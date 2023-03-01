from specutils import Spectrum1D
from astropy import units as u
import copy
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate, integrate
import matplotlib as mpl

from pahfit.features.util import bounded_is_fixed
from pahfit.component_models import BlackBody1D, Drude1D, S07_attenuation
from pahfit.helpers import find_packfile
from pahfit.features import Features
from pahfit.base import PAHFITBase
from pahfit import instrument
from pahfit.errors import PAHFITModelError


class Model:
    """This class acts as the main API for PAHFIT.

    The users deal with model objects, of which the state is modified
    during initalization, initial guessing, and fitting. What the model
    STORES is a description of the physics: what features are there and
    what are their properties, regardless of the instrument with which
    those features are observed. The methods provided by this class,
    form the connection between those physics, and what is observed.
    During fitting and plotting, those physics are converted into a
    model for the observation, by applying instrumental parameters from
    the instrument.py module.

    The main thing that defines a model, is the features table, loaded
    from a YAML file given to the constructor. After construction, the
    Model can be edited by accessing the stored features table directly.
    Changing numbers in this table, is allowed, and the updated numbers
    will be reflected when the next fit or initial guess happens. At the
    end of these actions, the fit or guess results are stored in the
    same table.

    The model can be saved.

    The model can be copied.

    Attributes
    ----------
    features : Features
        Instance of the Features class. Can be edited on-the-fly.
        Non-breaking behavior by the user is expected. Changes will be
        reflected at the next fit, guess, or plot call.

    """

    def __init__(self, features: Features):
        """
        Parameters
        ----------
        features: Features
            Features table.

        """
        self.features = features

        # If features table does not originate from a previous fit, and
        # hence has no unit yet, we initialize it as an empty dict.
        if "user_unit" not in self.features.meta:
            self.features.meta["user_unit"] = {}

        # store fit_info dict of last fit
        self.fit_info = None

    @classmethod
    def from_yaml(cls, pack_file):
        """
        Generate feature table from YAML file.

        Parameters
        ----------
        pack_file : str
            Path to YAML file, or name of one of the default YAML files.

        Returns
        -------
        Model instance

        """
        features = Features.read(pack_file)
        return cls(features)

    @classmethod
    def from_saved(cls, saved_model_file):
        """
        Parameters
        ----------
        saved_model_file : str
           Path to file generated by Model.save()

        Returns
        -------
        Model instance
        """
        # features.read automatically switches to astropy table reader.
        # Maybe needs to be more advanced here in the future.
        features = Features.read(saved_model_file, format="ascii.ecsv")
        return cls(features)

    def save(self, fn, **write_kwargs):
        """Save the model to disk.

        Only ECSV supported for now. Models saved this way can be read
        back in, with metadata.

        TODO: store details about the fit results somehow. Uncertainties
        (covariance matrix) should be retrievable. Use Table metadata?

        Parameters
        ----------
        fn : file name

        **write_kwargs : kwargs passed to astropy.table.Table.write

        """
        if fn.split(".")[-1] != "ecsv":
            raise NotImplementedError("Only ascii.ecsv is supported for now")

        self.features.write(fn, format="ascii.ecsv", **write_kwargs)

    def _status_message(self):
        out = "Model features ("
        if self.fit_info is None:
            out += "not "
        out += "fitted)\n"
        return out

    def __repr__(self):
        return self._status_message() + self.features.__repr__()

    def _repr_html_(self):
        return self._status_message() + self.features._repr_html_()

    def guess(
        self,
        spec: Spectrum1D,
        redshift=None,
        calc_line_fwhm=True,
        line=True,
        dust_feature=True,
    ):
        """Make an initial guess of the physics, based on the given
        observational data.

        Parameters
        ----------
        spec : Spectrum1D
            1D (not 2D or 3D) spectrum object, containing the
            observational data. (TODO: should support list of spectra,
            for the segment-based joint fitting). Initial guess will be
            based on the flux in this spectrum.

            spec.meta['instrument'] : str or list of str
                Qualified instrument name, see instrument.py. This will
                determine what the line widths are, when going from the
                features table to a fittable/plottable model.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model.

            If None, will be taken from spec.redshift

        Returns
        -------
        Nothing, but internal feature table is updated.

        """
        # parse spectral data
        self.features.meta["user_unit"]["flux"] = spec.flux.unit
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        _, _, _, xz, yz, _ = self._convert_spec_data(spec, z)
        wmin = min(xz)
        wmax = max(xz)

        # simple linear interpolation function for spectrum
        sp = interpolate.interp1d(xz, yz)

        # we will repeat this loop logic several times
        def loop_over_non_fixed(kind, parameter, estimate_function, force=False):
            for row_index in np.where(self.features["kind"] == kind)[0]:
                row = self.features[row_index]
                if not bounded_is_fixed(row[parameter]) or force:
                    guess_value = estimate_function(row)
                    # print(f"{row['name']}: setting {parameter} to {guess_value}")
                    self.features[row_index][parameter][0] = guess_value

        # guess starting point of bb
        def starlight_guess(row):
            bb = BlackBody1D(1, row["temperature"][0])
            w = wmin + 0.1  # the wavelength used to compare
            if w < 5:
                # wavelength is short enough to not have numerical
                # issues. Evaluate both at w.
                amp_guess = sp(w) / bb(w)
            else:
                # wavelength too long for stellar BB. Evaluate BB at
                # 5 micron, and spectrum data at minimum wavelength.
                wsafe = 5
                amp_guess = sp(w) / bb(wsafe)

            return amp_guess

        loop_over_non_fixed("starlight", "tau", starlight_guess)

        # count number of blackbodies in the model
        nbb = len(self.features[self.features["kind"] == "dust_continuum"])

        def dust_continuum_guess(row):
            temp = row["temperature"][0]
            fmax_lam = 2898.0 / temp
            bb = BlackBody1D(1, temp)
            if fmax_lam >= wmin and fmax_lam <= wmax:
                w = fmax_lam
                flux_ref = sp(w)
            elif fmax_lam > wmax:
                w = wmax
                flux_ref = yz[np.argmax(xz)]
            else:
                w = wmin
                flux_ref = yz[np.argmin(xz)]

            amp_guess = flux_ref / bb(w)
            return amp_guess / nbb

        loop_over_non_fixed("dust_continuum", "tau", dust_continuum_guess)

        def line_fwhm_guess(row):
            w = row["wavelength"][0]
            if not instrument.within_segment(w, inst):
                return 0

            fwhm = instrument.fwhm(inst, w, as_bounded=True)[0][0]
            return fwhm

        def amp_guess(row, fwhm):
            w = row["wavelength"][0]
            if not instrument.within_segment(w, inst):
                return 0

            factor = 1.5
            wmin = w - factor * fwhm
            wmax = w + factor * fwhm
            xz_window = np.logical_and(wmin < xz, xz < wmax)
            xpoints = xz[xz_window]
            ypoints = yz[xz_window]
            if np.count_nonzero(xz_window) >= 2:
                # difference between flux in window and flux around it
                power_guess = integrate.trapezoid(yz[xz_window], xz[xz_window])
                # subtract continuum estimate, but make sure we don't go negative
                continuum = (ypoints[0] + ypoints[-1]) / 2 * (xpoints[-1] - xpoints[0])
                if continuum < power_guess:
                    power_guess -= continuum
            else:
                power_guess = 0

            return power_guess / fwhm

        # calc line amplitude using instrumental fwhm and integral over data
        if line:
            loop_over_non_fixed(
                "line", "power", lambda row: amp_guess(row, line_fwhm_guess(row))
            )
        # set the fwhms in the features table requested
        if calc_line_fwhm:
            loop_over_non_fixed("line", "fwhm", line_fwhm_guess, force=True)

        def df_amp_guess(row):
            w = row["wavelength"][0]
            fwhm = row["fwhm"][0]
            if not (
                instrument.within_segment(w - fwhm, inst)
                or instrument.within_segment(w + fwhm, inst)
            ):
                return 0

            drude = Drude1D(amplitude=1, x_0=w, fwhm=fwhm)
            return sp(w) / drude(w)

        # for dust features, the fwhm is available in the table already
        if dust_feature:
            loop_over_non_fixed("dust_feature", "power", df_amp_guess)

    @staticmethod
    def _convert_spec_data(spec, z):
        """
        Turn astropy quantities stored in Spectrum1D into fittable
        numbers.

        Also corrects for redshift.

        Returns
        -------
        x, y, unc: wavelength in micron, flux, uncertainty

        xz, yz, uncz: wavelength in micron, flux, uncertainty
            corrected for redshift
        """
        x = spec.spectral_axis.to(u.micron).value
        y = spec.flux.value
        unc = spec.uncertainty.array

        # transform observed wavelength to "physical" wavelength
        xz = x / (1 + z)  # wavelength shorter
        yz = y * (1 + z)  # energy higher
        uncz = unc * (1 + z)  # uncertainty scales with flux
        return x, y, unc, xz, yz, uncz

    def fit(
        self,
        spec: Spectrum1D,
        redshift=None,
        maxiter=1000,
        verbose=True,
        use_instrument_fwhm=True,
    ):
        """Fit the observed data.

        The model setup is based on the features table and instrument
        specification.

        The last fit results can accessed through the variable
        model.astropy_result. The results are also stored back to the
        model.features table.

        CAVEAT: any features that do not overlap with the data range
        will not be included in the model, for performance and numerical
        stability. Their values in the features table will be left
        untouched.

        Parameters
        ----------
        spec : Spectrum1D
            1D (not 2D or 3D) spectrum object, containing the
            observational data. (TODO: should support list of spectra,
            for the segment-based joint fitting). Initial guess will be
            based on the flux in this spectrum.

            spec.meta['instrument'] : str or list of str
                Qualified instrument name, see instrument.py. This will
                determine what the line widths are, when going from the
                features table to a fittable/plottable model.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model.

            If None, will be taken from spec.redshift

        maxiter : int
            maximum number of fitting iterations

        verbose : boolean
            set to provide screen output

        use_instrument_fwhm : bool
            Use the instrument model to calculate the fwhm of the
            emission lines, instead of fitting them, which is the
            default behavior. This can be set to False to set the fwhm
            manually using the value in the science pack. If False and
            bounds are provided on the fwhm for a line, the fwhm for
            this line will be fit to the data.

        """
        # parse spectral data
        self.features.meta["user_unit"]["flux"] = spec.flux.unit
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        x, _, _, xz, yz, uncz = self._convert_spec_data(spec, z)

        # save these as part of the model (will be written to disk too)
        self.features.meta["redshift"] = inst
        self.features.meta["instrument"] = z

        # check if observed spectrum is compatible with instrument model
        instrument.check_range([min(x), max(x)], inst)

        # weigths
        w = 1.0 / uncz

        # clean, because astropy does not like nan
        mask = np.isfinite(xz) & np.isfinite(yz) & np.isfinite(w)

        # construct model and perform fit
        astropy_model = self._construct_astropy_model(inst, z, use_instrument_fwhm)
        fit = LevMarLSQFitter(calc_uncertainties=True)
        self.astropy_result = fit(
            astropy_model,
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

        self._parse_astropy_result(self.astropy_result)

    def info(self):
        """Print out the last fit results."""
        print(self.astropy_result)

    def plot(
        self,
        spec=None,
        redshift=None,
        use_instrument_fwhm=False,
        label_lines=False,
        **errorbar_kwargs,
    ):
        """Plot model, and optionally compare to observational data.

        Parameters
        ----------
        spec : Spectrum1D
            Observational data. Does not have to be the same data that
            was used for guessing or fitting.

        redshift : float
            Redshift used to shift from the physical model, to the
            observed model.

            If None, will be taken from spec.redshift

        """
        inst, z = self._parse_instrument_and_redshift(spec, redshift)
        _, _, _, xz, yz, uncz = self._convert_spec_data(spec, z)
        # total model
        model = self._construct_astropy_model(
            inst, z, use_instrument_fwhm=use_instrument_fwhm
        )
        enough_samples = max(10000, len(spec.wavelength))
        x_mod = np.logspace(np.log10(min(xz)), np.log10(max(xz)), enough_samples)

        fig, axs = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # spectrum and best fit model
        ax = axs[0]
        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.minorticks_on()
        ax.tick_params(
            axis="both", which="major", top="on", right="on", direction="in", length=10
        )
        ax.tick_params(
            axis="both", which="minor", top="on", right="on", direction="in", length=5
        )

        ext_model = None
        has_att = "attenuation" in self.features["kind"]
        has_abs = "absorption" in self.features["kind"]
        if has_att:
            row = self.features[self.features["kind"] == "attenuation"][0]
            tau = row["tau"][0]
            ext_model = S07_attenuation(tau_sil=tau)(x_mod)

        if has_abs:
            raise NotImplementedError(
                "plotting absorption features not implemented yet"
            )

        if has_att or has_abs:
            ax_att = ax.twinx()  # axis for plotting the extinction curve
            ax_att.tick_params(which="minor", direction="in", length=5)
            ax_att.tick_params(which="major", direction="in", length=10)
            ax_att.minorticks_on()
            ax_att.plot(x_mod, ext_model, "k--", alpha=0.5)
            ax_att.set_ylabel("Attenuation")
            ax_att.set_ylim(0, 1.1)
        else:
            ext_model = np.ones(len(x_mod))

        # Define legend lines
        Leg_lines = [
            mpl.lines.Line2D([0], [0], color="k", linestyle="--", lw=2),
            mpl.lines.Line2D([0], [0], color="#FE6100", lw=2),
            mpl.lines.Line2D([0], [0], color="#648FFF", lw=2, alpha=0.5),
            mpl.lines.Line2D([0], [0], color="#DC267F", lw=2, alpha=0.5),
            mpl.lines.Line2D([0], [0], color="#785EF0", lw=2, alpha=1),
            mpl.lines.Line2D([0], [0], color="#FFB000", lw=2, alpha=0.5),
        ]

        cont_model = None
        if "dust_continuum" in self.features["kind"]:
            cont_model = self.sub_model(inst, z, kind="dust_continuum")
            cont_y = cont_model(x_mod)
            # one plot for every component
            for c in cont_model:
                ax.plot(x_mod, c(x_mod) * ext_model, "#FFB000", alpha=0.5)
            # plot for total continuum?

        if "starlight" in self.features["kind"]:
            star_cont_model = self.sub_model(inst, z, kind="starlight")
            if cont_model is not None:
                cont_model += star_cont_model
            else:
                cont_model = star_cont_model

        # total continuum
        ax.plot(x_mod, cont_y * ext_model, "#785EF0", alpha=1)

        if "dust_feature" in self.features["kind"]:
            # now plot the dust bands and lines
            features_model = self.sub_model(inst, z, kind="dust_feature")
            for c in features_model:
                ax.plot(
                    x_mod,
                    (cont_y + c(x_mod)) * ext_model,
                    "#648FFF",
                    alpha=0.5,
                )

        if "line" in self.features["kind"]:
            lines_model = self.sub_model(
                inst, z, kind="line", use_instrument_fwhm=use_instrument_fwhm
            )
            for c in lines_model:
                ax.plot(
                    x_mod,
                    (cont_y + c(x_mod)) * ext_model,
                    "#DC267F",
                    alpha=0.5,
                )
                if label_lines:
                    w = c.mean.value
                    ax.text(
                        w,
                        model(w),
                        c.name,
                        va="center",
                        ha="center",
                        rotation="vertical",
                        bbox=dict(facecolor="white", alpha=0.75, pad=0),
                    )

        ax.plot(x_mod, model(x_mod), "#FE6100", alpha=1)

        # data
        default_kwargs = dict(
            fmt="o",
            markeredgecolor="k",
            markerfacecolor="none",
            ecolor="k",
            elinewidth=0.2,
            capsize=0.5,
            markersize=6,
        )

        ax.errorbar(xz, yz, yerr=uncz, **(default_kwargs | errorbar_kwargs))

        ax.set_ylim(0)
        ax.set_ylabel(r"$\nu F_{\nu}$")

        ax.legend(
            Leg_lines,
            [
                "S07_attenuation",
                "Spectrum Fit",
                "Dust Features",
                r"Atomic and $H_2$ Lines",
                "Total Continuum Emissions",
                "Continuum Components",
            ],
            prop={"size": 10},
            loc="best",
            facecolor="white",
            framealpha=1,
            ncol=3,
        )

        # residuals, lower sub-figure
        res = yz - model(xz)
        std = np.std(res)
        ax = axs[1]

        ax.set_yscale("linear")
        ax.set_xscale("log")
        ax.tick_params(
            axis="both", which="major", top="on", right="on", direction="in", length=10
        )
        ax.tick_params(
            axis="both", which="minor", top="on", right="on", direction="in", length=5
        )
        ax.minorticks_on()

        # Custom X axis ticks
        ax.xaxis.set_ticks(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30, 40]
        )

        ax.axhline(0, linestyle="--", color="gray", zorder=0)
        ax.plot(
            xz,
            res,
            "ko-",
            fillstyle="none",
            zorder=1,
            markersize=errorbar_kwargs.get("markersize", None),
            alpha=errorbar_kwargs.get("alpha", None),
            linestyle="none",
        )
        scalefac_resid = 2
        ax.set_ylim(-scalefac_resid * std, scalefac_resid * std)
        ax.set_xlim(np.floor(np.amin(xz)), np.ceil(np.amax(xz)))
        ax.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax.set_ylabel("Residuals [%]")

        # scalar x-axis marks
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        fig.subplots_adjust(hspace=0)
        return fig

    def copy(self):
        """Copy the model.

        Main use case: use this model as a parent model for more
        fits.

        Currently uses copy.deepcopy. We should do something smarter if
        we run into memory problems or sluggishness.

        Returns
        -------
        model_copy : Model
        """
        # A standard deepcopy works fine!
        return copy.deepcopy(self)

    def tabulate(
        self,
        instrumentname,
        redshift=0,
        wavelengths=None,
        feature_mask=None,
    ):
        """Tabulate model flux on a wavelength grid, and export as Spectrum1D

        The flux unit will be the same as the last fitted spectrum, or
        dimensionless if the model is tabulated before being fit.

        Parameters
        ----------
        wavelengths : Spectrum1D or array-like
            Wavelengths in micron in the observed frame. Will be
            multiplied with 1/(1+z) if redshift z is given, so that the
            model is evaluated in the rest frame as intended. If a
            Spectrum1D is given, wavelengths.spectral_axis will be
            converted to micron and then used as wavelengths.

        instrumentname : str or list of str
            Qualified instrument name, see instrument.py. This
            determines the wavelength range of features to be included.
            The FWHM of the unresolved lines will be determined by the
            value in the features table, instead of the instrument. This
            allows us to visualize the fitted line widths in the
            spectral overlap regions.

        redshift : float
            The redshift is needed to evaluate the flux model at the
            right rest wavelengths.

        feature_mask : array of bool of length len(features)
            Mask used to select specific rows of the feature table. In
            most use cases, this mask can be made by applying a boolean
            operation to a column of self.features, e.g.
            model.features['wavelength'] > 8.5

        Returns
        -------
        model_spectrum : Spectrum1D
            The flux model, evaluated at the given wavelengths, packaged
            as a Spectrum1D object.
        """
        # apply feature mask, make sub model, and set up functional
        if feature_mask is not None:
            features_copy = self.features.copy()
            features_to_use = features_copy[feature_mask]
        else:
            features_to_use = self.features
        alt_model = Model(features_to_use)

        # Always use the current FWHM here (use_instrument_fwhm would
        # overwrite the value in the instrument overlap regions!)
        flux_function = alt_model._construct_astropy_model(
            instrumentname, redshift, use_instrument_fwhm=False
        )

        # decide which wavelength grid to use
        if wavelengths is None:
            ranges = instrument.wave_range(instrumentname)
            wmin = min(r[0] for r in ranges)
            wmax = max(r[1] for r in ranges)
            wfwhm = instrument.fwhm(instrumentname, wmin, as_bounded=True)[0, 0]
            wav = np.arange(wmin, wmax, wfwhm / 2) * u.micron
        elif isinstance(wavelengths, Spectrum1D):
            wav = wavelengths.spectral_axis.to(u.micron)
        else:
            # any other iterable will be accepted and converted to array
            wav = np.asarray(wavelengths) * u.micron

        # shift the "observed wavelength grid" to "physical wavelength grid"
        wav /= 1 + redshift
        flux_values = flux_function(wav.value)

        # apply unit stored in features table (comes from from last fit
        # or from loading previous result from disk)
        if "flux" not in self.features.meta["user_unit"]:
            flux_quantity = flux_values * u.dimensionless_unscaled
        else:
            flux_quantity = flux_values * self.features.meta["user_unit"]["flux"]

        return Spectrum1D(spectral_axis=wav, flux=flux_quantity)

    def _kludge_param_info(self, instrumentname, redshift, use_instrument_fwhm=True):
        param_info = PAHFITBase.parse_table(self.features)
        # edit line widths and drop lines out of range

        # h2_info
        param_info[2] = PAHFITBase.update_dictionary(
            param_info[2],
            instrumentname,
            update_fwhms=use_instrument_fwhm,
            redshift=redshift,
        )
        # ion_info
        param_info[3] = PAHFITBase.update_dictionary(
            param_info[3],
            instrumentname,
            update_fwhms=use_instrument_fwhm,
            redshift=redshift,
        )
        # abs_info
        param_info[4] = PAHFITBase.update_dictionary(
            param_info[4], instrumentname, redshift
        )

        return param_info

    def _backport_param_info(self, param_info):
        """Convert param_info to values in features table.

        Temporary hack to make the new system compatible with the old system.

        TODO: if we remove the param_info stuff entirely, we won't need this

        """
        # unfortunately, there is no implementation for this, even in
        # the original code. That one goes straight from astropy model
        # to table... But we can do a kludge here: convert to model
        # first, and then back to table.
        astropy_model = PAHFITBase.model_from_param_info(param_info)
        self._parse_astropy_result(astropy_model)

    def _construct_astropy_model(
        self, instrumentname, redshift, use_instrument_fwhm=True
    ):
        """Convert the features table into a fittable model.

        Some nuances in the behavior
        - If a line has a fwhm set, it will be ignored, and replaced by
          the calculated fwhm provided by the instrument model.
        - If a line has been masked by _parse_astropy_result, and this
          function is called again, those masks will be ignored, as the
          data range might have changed.

        TODO: Make sure the features outside of the data range are
        removed. The instrument-based feature check is done in
        _kludge_param_info(), but the observational data might only
        cover a part of the instrument range.

        """
        param_info = self._kludge_param_info(
            instrumentname, redshift, use_instrument_fwhm
        )
        return PAHFITBase.model_from_param_info(param_info)

    def _parse_astropy_result(self, astropy_model):
        """Store the result of the astropy fit into the features table.

        Every relevant value inside the astropy model, is written to the
        right position in the features table.

        For the unresolved lines, the widths are calculated by the
        instrument model, or fitted when these lines are in a spectral
        overlap region. The calculated or fitted result is written to
        the fwhm field of the table. When a new model is constructed
        from the features table, this fwhm value will be ignored.

        For features that do not correspond to the data range, all
        parameter values will be masked. Their numerical values remain
        accessible by '.data' on the masked entity. This way, We still
        keep their parameter values around (as opposed to removing the
        rows entirely). When data with a larger range are passed for
        another fitting call, those features can be unmasked if
        necessary.

        """
        if len(self.features) < 2:
            # Plotting and tabulating works fine, but the code below
            # will not work with only one component. This can be
            # addressed later, when the internal API is made agnostic of
            # the fitting backend (astropy vs our own).
            raise PAHFITModelError("Fit with fewer than 2 components not allowed!")

        # Some translation rules between astropy model components and
        # feature table names and values.

        # these have the same value but different (fixed) names
        param_name_equivalent = {
            "temperature": "temperature",
            "fwhm": "fwhm",
            "x_0": "wavelength",
            "mean": "wavelength",
            "tau_sil": "tau",
        }

        def param_conversion(features_kind, param_name, param_value):
            # default conversion
            if param_name in param_name_equivalent:
                new_name = param_name_equivalent[param_name]
                new_value = param_value
            # certain types of features use tau instead of amplitude
            elif param_name == "amplitude":
                if features_kind in ["starlight", "dust_continuum", "absorption"]:
                    new_name = "tau"
                else:
                    new_name = "power"
                new_value = param_value
            # convert stddev to fwhm
            elif param_name == "stddev":
                new_name = "fwhm"
                new_value = param_value * 2.355
            else:
                raise NotImplementedError(
                    f"no conversion rule for model parameter {param_name}"
                )
            return new_name, new_value

        # Go over all features.
        for row in self.features:
            name = row["name"]
            if name in astropy_model.submodel_names:
                # undo any previous masking that might have occured
                self.features.unmask_feature(name)

                # copy or translate, and store the parameters
                component = astropy_model[name]
                for param_name in component.param_names:
                    param_value = getattr(component, param_name).value
                    col_name, col_value = param_conversion(
                        row["kind"], param_name, param_value
                    )
                    row[col_name][0] = col_value

                # for the unresolved lines, indicate when the line fwhm was made non-fixed
                if row["kind"] == "line" and col_name == "fwhm":
                    row["fwhm"].mask[1:] = component.fixed[param_name]
            else:
                # signal that it was not fit by masking the feature
                self.features.mask_feature(name)

    @staticmethod
    def _parse_instrument_and_redshift(spec, redshift):
        """Small utility to grab instrument and redshift from either
        Spectrum1D metadata, or from arguments.

        """
        # the rest of the implementation doesn't like Quantity...
        z = spec.redshift.value if redshift is None else redshift
        if z is None:
            # default of spec.redshift is None!
            z = 0

        inst = spec.meta["instrument"]
        if inst is None:
            raise PAHFITModelError("No instrument! Please set spec.meta['instrument'].")

        return inst, z
