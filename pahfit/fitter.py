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
from abc import ABC, abstractmethod


class Fitter(ABC):
    """Abstract base class for interal Fitter API.

    A couple of notes on the implementation of subclasses
    - All shared methods should have the same arguments, enforced by
      this abstract class.
    - Any API-specific options preferably go into the constructor of the
      subclass, although some general-purpose dictionaries could also be
      used if absolutely necessary.
    """

    @abstractmethod
    def clear(self):
        """Reset model.

        After reset, register_* and finalize_model can be used again.

        """
        pass

    @abstractmethod
    def components(self):
        """Return list of features.

        Only works after finalize_model(). Will return the names passed
        using the register functions.

        """
        pass

    @abstractmethod
    def finalize_model(self):
        """Process the registered features and prepare for fitting.

        The register functions below allow adding individual features.
        The exact implementation of how features are added, and
        finalized in to a single fittable model, depend on the
        underlying implementation.

        """
        pass

    @abstractmethod
    def register_starlight(self, name, temperature, tau):
        """Register a starlight feature.

        The exact representation depends on the implementation, but the
        meaning of the parameters should be equivalent.

        Parameters
        ----------
        name : str
            Unique name. Will be used to allow retrieval of the results
            after the fitting.

        temperature : array of size 3
            Blackbody temperature, given as [value, lower_bound,
            upper_bound]. The bounds assume the same system as the
            features table: if they are masked, the parameter will be
            fixed, while +inf or -inf mean unbounded.

        tau : array of size 3
            Analogously, used as power.

        """
        pass

    @abstractmethod
    def register_dust_continuum(self, name, temperature, tau):
        """Register a dust continuum feature."""
        pass

    @abstractmethod
    def register_line(self, name, power, wavelength, fwhm):
        """Register an emission line feature.

        Typically a Gaussian profile.

        """
        pass

    @abstractmethod
    def register_dust_features(self, name, power, wavelength, fwhm):
        """Register a dust feature.

        Typically a Drude profile.

        """
        pass

    @abstractmethod
    def register_attenuation(self, name, tau):
        """Register the S07 attenuation component.

        Other types of attenuation might be possible in the future. Is
        multiplicative.

        """
        pass

    @abstractmethod
    def register_absorption(self, name, tau, wavelength, fwhm):
        """Register an absorption feature.

        Typically a Drude profile. Is multiplicative.

        """
        pass

    @abstractmethod
    def evaluate_model(self, xz):
        """Evaluate the model at the given wavelengths.

        Parameters
        ----------
        xz : array
            Rest frame wavelengths in micron

        Returns
        -------
        yz : array
            Rest frame flux in internal units

        """
        pass

    @abstractmethod
    def fit(self, xz, yz, uncz, maxiter=1000):
        """Perform the fit using the framework of the subclass.

        Fitter is unit agnostic, and deals with the numbers the Model
        tells it to deal with. In practice, the input spectrum is
        expected to be in internal units, and corrected for redshift
        (models operate in the rest frame).

        After the fit, the results can be retrieved via get_result().

        Parameters
        ----------
        xz : array
            Rest frame wavelengths in micron

        yz : array
            Rest frame flux in internal units.

        uncz : array
            Uncertainty on rest frame flux. Same units as yz.

        """
        pass

    @abstractmethod
    def get_result(self, feature_name):
        """Retrieve results from underlying model after fit.

        Parameters
        ----------
        component_name : str
            One of the names provided to any of the register_*() calls
            made during setup. See also Fitter.components().

        Returns
        -------
        dict : parameters according to the PAHFIT definitions. Keys are
        the same as the function signature of the relevant register
        function. Values are in the same format as Features, and can
        therefore be directly filled in.

        e.g. {'name': 'line0', 'power': value, 'fwhm': value, 'wavelength'}

        """
        pass


class AstropyFitter(Fitter):
    """Astropy fitting implementation using hypothetical internal API.

    Once we actually define the internal API, there will be an abstract
    base class called "Fitter". For now, this draft was written with
    what is needed, and the Fitter class will be set up later to reflect
    the minimal needs.

    The main functionalities of a Fitter subclass:
    1. Convert the numbers that are in the Features table to a fittable
       model configuration for a certain framework. The details of the
       fitting framework are hidden behind the respective subclass.
    2. Fit the model to the spectrum without any additional assumptions.
       The Fitter will fit the given data using the given model without
       thinking about redshift, units, instrumental effects.)
    3. Retrieve the fitted quantities, which are the values that were
       passed during step 1. When fit result uncertainties are
       implemented, they will also need to be retrieved through this
       API.
    4. Access to the evaluation of the underlying model (again with no
       assumptions like in step 2.).

    The AstropyFitter backend translate the above needs for PAHFIT into
    actions that involve an astropy-based CompoundModel. Some details
    for the above tasks are given in the documentation of the respective
    functions.

    For the model setup, multiple functions are used, so a few notes are
    provided here. There is one function per type of component supported
    by PAHFIT, and the arguments of these functions will ask for
    different "standard" PAHFIT numbers, i.e. those from the Features
    table. After all the components have been added, the
    finalize_model() function can be called to finish setting up the
    internal astropy model. After this has finished, fit() can be
    called to apply the model and the astropy fitter to the data.

    These functions have the same signature between all Fitter
    implementations, so that the Model class can use a single
    implementation to set up the Fitter. The Model has access to the
    Features table and the instrument model, and needs to set up Fitter
    with the correct initial values, bounds, and "fixed" flags (e.g.
    setting a fixed FWHM based on the instrument for the lines).

    Multi-segment fitting was not implemented for this subclass, but
    here is a sketch for how it could work:
    1. During set up, an extra argument is passed, indicating the
       segment number. Each segment gets a different list of components
       to use.
    2. During finalize the joint model / something else for joint
       fitting is set up. Alternatively, all the separate models are
       constructed, and then a set of "unique" parameters is gathered.
       An appropriate name would be the "parameter union" (as it is like
       taking the union of the parameters sets for the individual
       segments).
    3. During the fitting, the objective function (chi2) is the sum of
       the objective functions for the individual models. The arguments
       of this objective function, are the unique parameters that were
       derived above. At every fitting step, these parameters will
       change. Inside the objective function, these changes should be
       propagated to the individual models, after which they can be
       evaluated again. All the fitting algorithm needs, is the
       parameter space, and the objective function value.

    Alternative idea, based on concatenated segments (not spectrally
    merged! Just a raw data concatenation).
    - concatenate x
    - concatenate y
    - bookkeep the boundaries between the concatenations
    - when evaluating, fill the concatenated model spectrum using a
      model that depends on the index (according to the boundaries that
      were set up)
    - Use the 'tied' option for parameters that should be equivalent
      (e.g. starlight in segment 1 should have same temperature as
      starlight in segment 2).

    """

    def __init__(self):
        """Construct a new fitter.

        After construction, use the register_* functions to start
        setting up a model, then call finalize_model().

        """
        self.clear()

    def clear(self):
        """Reset model.

        After reset, register_* and finalize_model can be used again.

        """
        self.additive_components = []
        self.multiplicative_components = []
        self.component_types = {}
        self.model = None

    def components(self):
        """Return list of component names.

        Only works after finalize_model().

        """
        if hasattr(self.model, "submodel_names"):
            return self.model.submodel_names
        else:
            # single-component edge case
            return [self.model.name]

    def finalize_model(self):
        """Sum the registered components into one CompoundModel.

        To be called after a series of "register_*()" calls, and before
        using the other functionality (fitting and evaluating).

        """
        if len(self.additive_components) > 1:
            self.model = sum(self.additive_components[1:], self.additive_components[0])
        elif len(self.additive_components) == 1:
            self.model = self.additive_components[0]
        else:
            raise PAHFITModelError("No components were set up for AstropyFitter!")

        for c in self.multiplicative_components:
            self.model *= c

    def _register_component(self, astropy_model_class, multiplicative=False, **kwargs):
        """Register any component in a generic way.

        Parameters
        ----------
        astropy_model_class : class
            Class symbol, e.g. Blackbody1D.

        multiplicative : bool
            During finalize_model(), all components with multiplicative
            == False will be added together, and then the model will be
            multiplied with the components with multiplicative == True.

        kwargs : dict
            Arguments for the astropy model constructor, including a
            unique value for "name". Should be generated with
            self._astropy_model_kwargs; the register* functions show how
            to do this for each type of feature.

        """
        if multiplicative:
            self.multiplicative_components.append(astropy_model_class(**kwargs))
        else:
            self.additive_components.append(astropy_model_class(**kwargs))

    def register_starlight(self, name, temperature, tau):
        """Register a BlackBody1D.

        Parameters
        ----------
        name : str
            Unique name.

        temperature : array of size 3
            Temperature for the blackbody, given as [value, lower_bound,
            upper_bound]. The bounds assume the same system as the
            features table, where masked == fixed, and +inf or -inf mean
            unbounded.

        tau : analogous. Used as amplitude.

        """
        self.component_types[name] = "starlight"
        kwargs = self._astropy_model_kwargs(
            name, ["temperature", "amplitude"], [temperature, tau]
        )
        self._register_component(BlackBody1D, **kwargs)

    def register_dust_continuum(self, name, temperature, tau):
        """Register a ModifiedBlackBody1D.

        Analogous. Temperature and tau are used as temperature and
        amplitude

        """
        self.component_types[name] = "dust_continuum"
        kwargs = self._astropy_model_kwargs(
            name, ["temperature", "amplitude"], [temperature, tau]
        )
        self._register_component(ModifiedBlackBody1D, **kwargs)

    def register_line(self, name, power, wavelength, fwhm):
        """Register a Gaussian1D.

        Analogous. Converts fwhm to stddev internally. Will be replaced
        by a Gaussian model that uses area instead of amplitude.

        """
        self.component_types[name] = "line"
        kwargs = self._astropy_model_kwargs(
            name, ["amplitude", "mean", "stddev"], [power, wavelength, fwhm / 2.355]
        )
        self._register_component(Gaussian1D, **kwargs)

    def register_dust_feature(self, name, power, wavelength, fwhm):
        """Register a Drude1D.

        Analogous. Will be replaced by AreaDrude.

        """
        self.component_types[name] = "dust_feature"
        kwargs = self._astropy_model_kwargs(
            name, ["amplitude", "x_0", "fwhm"], [power, wavelength, fwhm]
        )
        self._register_component(Drude1D, **kwargs)

    def register_attenuation(self, name, tau):
        """Register the S07 attenuation component.

        Analogous. Uses tau as tau_sil for S07_attenuation. Is
        multiplicative.

        """
        self.component_types[name] = "attenuation"
        kwargs = self._astropy_model_kwargs(name, ["tau_sil"], [tau])
        self._register_component(S07_attenuation, multiplicative=True, **kwargs)

    def register_absorption(self, name, tau, wavelength, fwhm):
        """Register an absorbing Drude1D component.

        Analogous. Is multiplicative.

        """
        self.component_types[name] = "absorption"
        kwargs = self._astropy_model_kwargs(
            name, ["tau", "x_0", "fwhm"], [tau, wavelength, fwhm]
        )
        self._register_component(att_Drude1D, multiplicative=True, **kwargs)

    def evaluate_model(self, xz):
        """Evaluate internal astropy model with its current parameters.

        Parameters
        ----------
        xz : array
            Rest frame wavelengths in micron

        Returns
        -------
        yz : array
            Rest frame flux in internal units
        """
        return self.model(xz)

    def fit(self, xz, yz, uncz, verbose=False, maxiter=10000):
        """Fit the internal model using the astropy fitter.

        The fitter class is unit agnostic, and deal with the numbers the
        Model tells it to deal with. Internal renormalizations could be
        good to consider, as long as any values are converted back to
        the original system before returning them. In practice, the
        input spectrum is expected to be in internal units, and orrected
        for redshift (models operate in the rest frame).

        After the fit, the results can be retrieved via get_result().

        Retrieval of uncertainties and fit details is yet to be
        implemented.

        Parameters
        ----------
        xz : array
            Rest frame wavelengths in micron

        yz : array
            Rest frame flux in internal units.

        uncz : array
            Uncertainty on rest frame flux. Same units as yz.

        """
        # clean, because astropy does not like nan
        w = 1 / uncz

        # make sure there are no zero uncertainties either
        mask = np.isfinite(xz) & np.isfinite(yz) & np.isfinite(w)

        fit = LevMarLSQFitter(calc_uncertainties=True)
        astropy_result = fit(
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

        self.model = astropy_result

    def get_result(self, component_name):
        """Retrieve results from astropy model component.

        Every relevant value inside the astropy model, need to be
        written to the right position in the features table. For some
        cases (amplitude/power, fwhm/stddev), conversions are necessary
        (generally the inverse conversions of what was done in the
        register function).

        Parameters
        ----------
        component_name : str
            One of the names provided to any of the register_*() calls
            made during setup. See also Fitter.components().

        Returns
        -------
        dict with Parameters according to the PAHFIT definitions.

        e.g. {'power': converted from amplitude, 'fwhm': converted from
        stddev, 'mean': wavelength}

        """
        if self.model is None:
            raise PAHFITModelError("Model not finalized yet.")

        if hasattr(self.model, "submodel_names"):
            component = self.model[component_name]
        else:
            # deals with edge case with single component, so is not
            # CompoundModel but normal single-component model.
            component = self.model

        c_type = self.component_types[component_name]
        if c_type == "starlight" or c_type == "dust_continuum":
            return {
                "temperature": component.temperature.value,
                "tau": component.amplitude.value,
            }
        elif c_type == "line":
            return {
                "power": component.amplitude.value,
                "wavelength": component.mean.value,
                "fwhm": component.stddev.value * 2.355,
            }
        elif c_type == "dust_feature":
            return {
                "power": component.amplitude.value,
                "wavelength": component.x_0.value,
                "fwhm": component.fwhm.value,
            }
        elif c_type == "attenuation":
            return {"tau": component.tau_sil.value}
        elif c_type == "absorption":
            return {
                "tau": component.tau.value,
                "wavelength": component.x_0.value,
                "fwhm": component.fwhm.value,
            }
        else:
            raise PAHFITModelError(f"Unsupported component type: {c_type}")

    @staticmethod
    def _astropy_model_kwargs(component_name, param_names, value_tuples):
        """Create kwargs for an astropy model constructor.

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
            Unique name for the component. Will later be used for
            indexing the components in the astropy model.

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
