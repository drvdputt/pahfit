import numpy as np
from scipy import interpolate
from astropy.modeling.physical_models import Drude1D
from astropy.modeling import Fittable1DModel
from astropy.modeling import Parameter
from astropy import constants
from astropy import units as u
from pahfit import units

__all__ = [
    "BlackBody1D",
    "ModifiedBlackBody1D",
    "S07_attenuation",
    "att_Drude1D",
    "PowerDrude1D",
    "PowerGaussian1D",
]

SQRT2PI = np.sqrt(2 * np.pi)
CMICRON = constants.c.to(u.micron / u.s).value


class PowerDrude1D(Fittable1DModel):
    """Drude profile with amplitude determined by power.

    This implementation is 'unitful' because 'power' is defined as
    integral over frequency, while the profile is evaluated as a
    function of wavelength. If we want the output y to have a certain
    unit(y), then the power parameter needs to have unit(y) * unit(c) /
    unit(x).

    The profile is a function of x (the wavelength), while the power is
    integral of profile over c / x. So the unit choice of c matters too.
    Example: if x_0 and fwhm are in micron, and the target flux is in
    u(flux), then the power parameter power will have unit u(flux) * Hz,
    e.g. mJy * Hz (if unit(c) was chosens as unit(x) Hz).

    In the implementation, we use the 'internal' units defined in
    pahfit.units to convert the input (power) and output (flux). We
    prevent ambiguities by enforcing a single unit for each parameter.
    There is one remaining problem: we don't know the flux units, so
    it's still ambiguous if power needs to be in intensity_power or
    flux_density_power units.

    An alternative approach, would be to do the fitting in Fnu(nu)
    space, instead of Fnu(lambda). In that case, we need an Fnu(nu)
    formulation of all the profiles and continuum functions we use.

    """

    power = Parameter(min=0.0)
    x_0 = Parameter(min=0.0)
    fwhm = Parameter(default=1, min=0.0)

    intensity_amplitude_factor = (
        (units.intensity_power * units.wavelength / (constants.c * np.pi))
        .to(units.intensity)
        .value
    )

    flux_density_amplitude_factor = (
        (units.flux_power * units.wavelength / (constants.c * np.pi))
        .to(units.flux_density)
        .value
    )

    @staticmethod
    def evaluate(x, power, x_0, fwhm):
        """Smith, et al. (2007) dust features model. Calculation is for
        a Drude profile (equation in section 4.1.4).

        The intensity profile as a function of wavelength is

        Inu(lambda) = (b * g**2) / ((lambda / x0 - x0 / lambda)**2 + g**2)

        With
        b = central intensity
        g = fwhm / x0
        x0 = central wavelength

        The integrated power (Fnu integrated over dnu) of the drude
        profile is

        P = (pi*c/2)*(b * g / x0)

        Which can be solved for the central intensity.

        b = (P * 2 * x0) / (pi * c * g) = 2P / (pi nu0 g).

        The output unit for the profile is unit(P) * Hz-1. The fitted
        value for P will be large, because of the nu0 factor in the
        denominator. We can make help the fitter by letting it deal more
        reasonable numbers. We do this by using 'power' as the fit
        parameter, which is the power in internal units, and is
        converted to P and then b in the implementation of this
        evaluation function.

        Parameters
        ----------
        power : float
        fwhm : float
        central intensity (x_0) : float

        """
        g = fwhm / x_0

        # amplitude in the right output unit
        # P = power * units.intensity_power
        # output_unit = units.intensity
        # lamb = x_0 * units.wavelength
        # b = (2 * P * lamb / (np.pi * constants.c * g)).to(output_unit).value
        # e.g. c = micron Hz -> b = flux unit = power Hz-1
        # so power unit needs to be flux unit Hz

        # use predetermined unit factor (already includes c, pi, and all units)
        b = 2 * power * x_0 / (np.pi * g) * PowerDrude1D.intensity_amplitude_factor
        return b * g**2 / ((x / x_0 - x_0 / x) ** 2 + g**2)


class PowerGaussian1D(Fittable1DModel):
    """Gaussian profile with amplitude derived from power.

    Implementation and caveats analogous to PowerDrude1D.

    The amplitude of a gaussian line of power P in per-wavelength units
    for the flux, is P / (stddev sqrt(2 pi)).

    Converting to per-frequency units, gives an amplitude of Fnu of
    A = P * lambda**2 / (c * stddev sqrt(2 pi))

    which we approximate here as
    A = P * mean**2 / (c * stddev sqrt(2 pi)).

    Constant conversion factor to put A in the right units:
    (unit(power) * unit(wavelength)**2 / (c * unit(wavelength))).to(unit.A)

    """

    power = Parameter(min=0.0)
    mean = Parameter()
    stddev = Parameter(default=1, min=0.0)

    intensity_amplitude_factor = (
        (
            units.intensity_power
            * (units.wavelength) ** 2
            / (constants.c * units.wavelength)
        )
        .to(units.intensity)
        .value
    )

    flux_density_amplitude_factor = (
        (units.flux_power * (units.wavelength) ** 2 / (constants.c * units.wavelength))
        .to(units.flux_density)
        .value
    )

    @staticmethod
    def evaluate(x, power, mean, stddev):
        """Evaluate F_nu(lambda) with a power.

        See class details for equations and unit notes."""

        # Astropy unit version
        # P = power * units.intensity_power
        # s = stddev * units.wavelength
        # m = mean * units.wavelength
        # output_unit = units.intensity
        # A = ((P * m**2) / (constants.c * s * SQRT2PI)).to(output_unit).value

        # Single factor version (probably faster than dealing with
        # astropy units every time). Factor c is already in it.
        A = (
            power
            * mean**2
            / (stddev * SQRT2PI)
            * PowerGaussian1D.intensity_amplitude_factor
        )

        return A * np.exp(-0.5 * np.square((x - mean) / stddev))


class BlackBody1D(Fittable1DModel):
    """
    A blackbody component.

    Current astropy BlackBody1D does not play well with Lorentz1D and Gauss1D
    maybe, need to check again, possibly a units issue
    """

    amplitude = Parameter()
    temperature = Parameter()

    @staticmethod
    def evaluate(x, amplitude, temperature):
        """ """
        return (
            amplitude
            * 3.97289e13
            / x**3
            / (np.exp(1.4387752e4 / x / temperature) - 1.0)
        )


class ModifiedBlackBody1D(BlackBody1D):
    """
    Modified blackbody with an emissivity propoportional to nu^2
    """

    @staticmethod
    def evaluate(x, amplitude, temperature):
        return BlackBody1D.evaluate(x, amplitude, temperature) * ((9.7 / x) ** 2)


class S07_attenuation(Fittable1DModel):
    """
    Smith, Draine, et al. (2007) kvt attenuation model calculation.
    Calculation is for a fully mixed geometrically model.
    Uses an extinction curve based on the silicate profiles from
    Kemper, Vriend, & Tielens (2004, apJ, 609, 826).
    Constructed as a weighted sum of two components: silicate profiles,
    and an exponent 1.7 power-law.

    Attenuation curve for a mixed case calculated from
    .. math::

        Att(x) = \\frac{1 - e^{-\\tau_{x}}}{\\tau_{x}}

    Parameters
    ----------
    kvt_amp : float
      amplitude
    """

    # Attenuation tau
    tau_sil = Parameter(description="kvt term: amplitude", default=1.0, min=0.0, max=10)

    @staticmethod
    def kvt(in_x):
        """
        Output the kvt extinction curve
        """
        # fmt: off
        kvt_wav = np.array([8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.7,
                            9.75, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0,
                            11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6,
                            12.7])

        kvt_int = np.array([.06, .09, .16, .275, .415, .575, .755, .895, .98,
                            .99, 1.0, .99, .94, .83, .745, .655, .58, .525,
                            .43, .35, .27, .20, .13, .09, .06, .045, .04314])
        # fmt: on

        # Extend kvt profile to shorter wavelengths
        if min(in_x) < min(kvt_wav):
            kvt_wav_short = in_x[in_x < min(kvt_wav)]
            kvt_int_short_tmp = min(kvt_int) * np.exp(
                2.03 * (kvt_wav_short - min(kvt_wav))
            )
            # Since kvt_int_shoft_tmp does not reach min(kvt_int),
            # we scale it to stitch it.
            kvt_int_short = kvt_int_short_tmp * (kvt_int[0] / max(kvt_int_short_tmp))

            spline_x = np.concatenate([kvt_wav_short, kvt_wav])
            spline_y = np.concatenate([kvt_int_short, kvt_int])
        else:
            spline_x = kvt_wav
            spline_y = kvt_int

        intfunc = interpolate.interp1d(spline_x, spline_y)
        in_x_spline = in_x[in_x < max(kvt_wav)]
        new_spline_y = intfunc(in_x_spline)

        nf = Drude1D(amplitude=0.4, x_0=18.0, fwhm=0.247 * 18.0)
        in_x_drude = in_x[in_x >= max(kvt_wav)]

        ext = np.concatenate([new_spline_y, nf(in_x_drude)])

        # Extend to ~2 um
        # assuming beta is 0.1
        beta = 0.1
        y = (1.0 - beta) * ext + beta * (9.7 / in_x) ** 1.7

        return y

    def evaluate(self, in_x, tau_si):
        if tau_si == 0.0:
            return np.full((len(in_x)), 1.0)
        else:
            tau_x = tau_si * self.kvt(in_x)
            return (1.0 - np.exp(-1.0 * tau_x)) / tau_x


class att_Drude1D(Fittable1DModel):
    """
    Attenuation components that can be parameterized by Drude profiles.
    """

    tau = Parameter()
    x_0 = Parameter()
    fwhm = Parameter()

    @staticmethod
    def evaluate(x, tau, x_0, fwhm):
        if tau == 0.0:
            return np.full((len(x)), 0.0)
        else:
            profile = Drude1D(amplitude=1.0, fwhm=fwhm, x_0=x_0)
            tau_x = tau * profile(x)
            return (1.0 - np.exp(-1.0 * tau_x)) / tau_x
