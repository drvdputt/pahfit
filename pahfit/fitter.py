from pahfit.errors import PAHFITModelError
from pahfit.component_models import (
    BlackBody1D,
    ModifiedBlackBody1D,
    Gaussian1D,
    Drude1D,
    S07_attenuation,
    att_Drude1D,
)
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter


class AstropyFitter:
    """This class is one implementation using the hypothetical internal API.

    Once we actually define the internal API, I can just rewrite a few
    things. For now, I will just write it in a way that makes sense to
    me.

    The main function: converting between the numbers that are in the
    Features table, and a multi-component astropy model. The latter is
    an implementation detail, and hidden behind this class. Just like
    JDT's custom fitter (which should be more performant, and is to be
    added soon?).

    The class manages an astropy model object, and exposes changes to
    that object (adding components) to the Model class by providing a
    set of functions. There is one function per type of component, and
    the function's arguments will ask for different numbers from the
    Features table. If these functions are the same between all "Fitter"
    implementations, then we can just write a loop in the Model class to
    set up the fitting object.

    Additional abstractions to be implemented include fitting the model,
    and the inverse of the constructor functions: reading the model
    output and returning numbers that can be put into the features
    table. After the fit, the Model class can then loop over the
    features table again, and fill in the numbers as necessary.

    Sketch for how multi-segment fitting could work:

    1. During set up, an extra argument is passed, indicating the
    segment number. Each segment gets a different list of components to
    use.

    2. During finalize the joint model / something else for joint
    fitting is set up (if using native astropy method?). Alternatively,
    all the separate models are constructed, and then a set of "unique"
    parameters is gathered. This is called the "parameter union" (as it
    is like taking the union of the parameters sets for the individual
    segments).

    3. During the fitting, the objective function (chi2) is the sum of
    the objective functions for the individual models. The arguments of
    this objective function, are the unique parameters that were derived
    above. At every fitting step, these parameters will change. Inside
    the objective function, these changes should be propagated to the
    individual models, after which they can be evaluated again. All the
    fitting algorithm needs, is the parameter space, and the objective
    function value.

    So conceptually simple, but bookkeeping and performance will be
    tricky.

    Another sudden idea: use concatenations!

    - concatenate x
    - concatenate y
    - bookkeep the boundaries between the concatenations
    - when evaluating, fill the concatenated model spectrum using a
      model that depends on the index (according to the boundaries that
      were set up)
    - This still has the problem of how to tie the parameters of the
      segments to the parameter union. Ideally, we would edit the values
      by adress, but don't know if the internals of the fitting
      algorithm would allow this.

    """

    def __init__(self):
        """Construct a new fitter.

        Set up has not happened yet.
        """
        self.clear()

    def clear(self):
        """Reset model, so a new one can be created."""
        self.additive_components = []
        self.multiplicative_components = []
        self.model = None

    def finalize_model(self):
        """Set up the compound model.

        To be called after calling a series of "register_*" functions.

        """
        if len(self.additive_components) > 1:
            self.model = sum(self.additive_components[1:], self.additive_components[0])
        elif len(self.additive_components) == 1:
            self.model = self.additive_components[0]
        else:
            PAHFITModelError("No components were set up for AstropyFitter!")

        for c in self.multiplicative_components:
            self.model *= c

    def _register_component(self, astropy_model_class, multiplicative=False, **kwargs):
        """Call constructor of astropy model and append to component list.

        kwargs should be generated with self._constructor_kwargs. The
        functions specialized for each type of feature know how to do
        this.

        """
        if multiplicative:
            self.multiplicative_components.append(astropy_model_class(**kwargs))
        else:
            self.additive_components.append(astropy_model_class(**kwargs))

    def register_starlight(self, name, temperature, tau):
        """Register a BlackBody1D component to be added to the astropy model.

        temperature and tau in internal units already, each should be a
        3-tuple (start, min, max), with the bounds set to None if the
        parameter is to be fixed.

        """
        kwargs = self._constructor_kwargs(
            name, ["temperature", "amplitude"], [temperature, tau]
        )
        self._register_component(BlackBody1D, **kwargs)

    def register_dust_continuum(self, name, temperature, tau):
        """Register a ModifiedBlackBody1D.

        Temperature and tau are used as temperature and amplitude

        """
        kwargs = self._constructor_kwargs(
            name, ["temperature", "amplitude"], [temperature, tau]
        )
        self._register_component(ModifiedBlackBody1D, **kwargs)

    def register_line(self, name, power, wavelength, fwhm):
        """Register a Gaussian1D.

        Converts fwhm to stddev internally

        Will later be replaced by a Gaussian model that uses area
        instead of amplitude.

        """
        kwargs = self._constructor_kwargs(
            name, ["amplitude", "mean", "stddev"], [power, wavelength, fwhm / 2.355]
        )
        self._register_component(Gaussian1D, **kwargs)

    def register_dust_feature(self, name, power, wavelength, fwhm):
        """Register a Drude1D.

        Will later be replaced by AreaDrude.

        """
        kwargs = self._constructor_kwargs(
            name, ["amplitude", "x_0", "fwhm"], [power, wavelength, fwhm]
        )
        self._register_component(Drude1D, **kwargs)

    def register_attenuation(self, name, tau):
        """Register the S07 attenuation component.

        Uses tau as tau_sil for S07_attenuation.

        """
        kwargs = self._constructor_kwargs(name, ["tau_sil"], [tau])
        self._register_component(S07_attenuation, multiplicative=True, **kwargs)

    def register_absorption(self, name, tau, wavelength, fwhm):
        """Register an absorbing Drude1D component."""
        kwargs = self._constructor_kwargs(
            name, ["tau", "x_0", "fwhm"], [tau, wavelength, fwhm]
        )
        self._register_component(att_Drude1D, multiplicative=True, **kwargs)

    @staticmethod
    def _constructor_kwargs(component_name, param_names, value_tuples):
        """Create kwargs for the astropy model constructor.

        This is a utility that deduplicates the logic for going from
        (value, min, max) tuples, to astropy model constructor keyword
        arguments as in the following example:

        AstropyModelClass(name="feature name",
                    param1=value1,
                    param2=value2,
                    bounds={param1: (min,max), param2:(min,max)},
                    fixed={param1: True, param2: False})

        The returned arguments are in a dict that looks as follows, and
        can be passed to the appropriate astropy model constructor using
        **kwargs.

        {"name": "feature name"
         param_name: double, ...,
         "bounds": {param_name: array of size 2, ...},
         "fixed": {param_name: True or False, ...}}

        Parameters:
        -----------

        component_name : str
            A name for the component. Will later be used for indexing
            the components in the Astropy model.

        param_names : list of str
            Names of the parameters for the astropy model, e.g.
            ["dust_feature1", "dust_feature2"]

        value_tuples : list of (array of size 3)
            One for each param name, each in the format of [value,
            min_bound, max_bound], i.e. in the format as stored in the
            Features table. This means that [value, masked, masked] will
            result in a fixed parameter.

        Returns
        -------

        dict : kwargs to be used in an astropy model constructor

        """
        # basic format of constructor parameters of astropy model
        kwargs = {"name": component_name, "bounds": {}, "fixed": {}}

        for param_name, value_tuple in zip(param_names, value_tuples):
            kwargs[param_name] = value_tuple[0]

            # Convention in the table: masked means fixed, infinity
            # means unbounded. So we set fixed argument to True when a
            # limit is masked. Also, astropy does not like numpy bools,
            # so we do this silly conversion.
            kwargs["fixed"][param_name] = True if value_tuple.mask[1] else False

            # For the limits, use 0 if fixed, the raw values if
            # variable, and None if variable but unbounded.
            limits = [0 if value_tuple.mask[i] else value_tuple[i] for i in range(1, 3)]
            kwargs["bounds"][param_name] = [None if np.isinf(x) else x for x in limits]

        return kwargs

    def fit(self, xz, yz, uncz, verbose=False, maxiter=10000):
        """Fit the astropy model using the astropy fitter.

        Ideally, the whole fitter class is unit agnostic. It should deal
        with the numbers the Model tells it to deal with. Internal
        renormalizations are allowed of course, as long as the results
        are converted back to the original system before returning them.

        In practice, let's follow the these guidelines:
        - The input is expected to be in internal units
        - Corrected for redshift (models operate in the rest frame).
        - Has no nan's or inf's
        - No zero uncertainties

        Parameters
        ----------
        xz : array
            Rest frame wavelengths in micron

        yz : array
            Rest frame flux in arbitrary units? Will determine the units of
            amplitude / power... So we should set the internal unit
            here.

        uncz : array
            Uncertaingty on rest frame flux. Same units as yz.

        """
        # clean, because astropy does not like nan
        w = 1 / uncz

        # make sure there are no zero uncertainties either
        mask = np.isfinite(xz) & np.isfinite(yz) & np.isfinite(w)

        fit = LevMarLSQFitter(calc_uncertainties=True)
        self.astropy_result = fit(
            self.model,
            xz[mask],
            yz[mask],
            weights=w[mask],
            maxiter=maxiter,
            epsilon=1e-10,
            acc=1e-10,
        )
        self.fit_info = fit.fit_info
        if verbose:
            print(fit.fit_info["message"])

    def component_param_values(self, name):
        """Do we also get multiple functions? Or just return a string
        that indicates the format? Or just a dict with the right
        parameters for the features table?

        Some pseudo-code for now:"""

        # get the component based on the name
        component = self.model["name"]

        # ! this does not work yet, need to translate the parameters again!
        return {
            param_name: param_value
            for param_name, param_value in zip(
                component.param_names, component.parameters
            )
        }
