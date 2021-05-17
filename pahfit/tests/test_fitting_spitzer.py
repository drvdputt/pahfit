import numpy as np

from pahfit.helpers import read_spectrum, initialize_model, fit_spectrum


def test_fitting_m101():

    # read in the spectrum
    spectrumfile = "M101_Nucleus_irs.ipac"
    obsdata = read_spectrum(spectrumfile)

    # setup the model
    packfile = "scipack_ExGal_SpitzerIRSSLLL.ipac"
    pmodel = initialize_model(packfile, obsdata, estimate_start=True)

    # fit the spectrum
    obsfit = fit_spectrum(obsdata, pmodel, maxiter=200)

    # fmt: off
    expvals = np.array([
        0.00000000e+00, 5.00000000e+03, 3.47599860e-08, 3.00000000e+02,
        4.95838415e-09, 2.00000000e+02, 0.00000000e+00, 1.35000000e+02,
        7.23019059e-05, 9.00000000e+01, 1.01195174e-03, 6.50000000e+01,
        1.46671071e-03, 5.00000000e+01, 0.00000000e+00, 4.00000000e+01,
        2.26196791e-01, 3.50000000e+01, 3.20506257e+00, 5.27000000e+00,
        1.79180000e-01, 1.78233271e+00, 5.70000000e+00, 1.99500000e-01,
        3.53244685e+01, 6.22000000e+00, 1.86600000e-01, 3.00299533e+00,
        6.69000000e+00, 4.68300000e-01, 7.81921883e+00, 7.42000000e+00,
        9.34920000e-01, 3.43758753e+01, 7.60000000e+00, 3.34400000e-01,
        3.33817792e+01, 7.85000000e+00, 4.16050000e-01, 6.84681209e+00,
        8.33000000e+00, 4.16500000e-01, 2.71687219e+01, 8.61000000e+00,
        3.35790000e-01, 6.56935555e-01, 1.06800000e+01, 2.13600000e-01,
        3.16704463e+01, 1.12300000e+01, 1.34760000e-01, 3.54974660e+01,
        1.13300000e+01, 3.62560000e-01, 8.53755133e+00, 1.19900000e+01,
        5.39550000e-01, 1.93502653e+01, 1.26200000e+01, 5.30040000e-01,
        7.58851547e+00, 1.26900000e+01, 1.64970000e-01, 7.91686821e+00,
        1.34800000e+01, 5.39200000e-01, 1.44924963e+00, 1.40400000e+01,
        2.24640000e-01, 7.38257692e+00, 1.41900000e+01, 3.54750000e-01,
        0.00000000e+00, 1.59000000e+01, 3.18000000e-01, 1.52027196e+01,
        1.64500000e+01, 2.30300000e-01, 2.72997411e+01, 1.70400000e+01,
        1.10760000e+00, 8.18088143e+00, 1.73750000e+01, 2.08500000e-01,
        2.91838398e+00, 1.78700000e+01, 2.85920000e-01, 4.48597325e+00,
        1.89200000e+01, 3.59480000e-01, 2.33160787e+01, 3.31000000e+01,
        1.65500000e+00, 8.67192733e-01, 5.52624727e+00, 1.12526539e-02,
        1.23767318e+01, 6.10984747e+00, 1.12526539e-02, 5.48881885e+00,
        6.89947186e+00, 1.13265497e-02, 1.46346475e+00, 8.03873301e+00,
        4.38292984e-02, 1.70893220e+00, 9.68075948e+00, 6.36942675e-02,
        0.00000000e+00, 1.23285000e+01, 6.36942675e-02, 1.10069789e+01,
        1.70022207e+01, 3.88847916e-02, 1.30109782e+01, 2.81707000e+01,
        2.16560510e-01, 2.19719193e+01, 6.98752346e+00, 1.23509488e-02,
        0.00000000e+00, 8.98627316e+00, 4.52465026e-02, 3.44832523e+00,
        1.05068705e+01, 2.12314225e-02, 3.10265440e+01, 1.28296867e+01,
        6.36942675e-02, 0.00000000e+00, 1.55050000e+01, 8.91719745e-02,
        3.27041174e+01, 1.87321438e+01, 6.86322478e-02, 0.00000000e+00,
        2.59600000e+01, 2.16560510e-01, 0.00000000e+00, 2.59390000e+01,
        2.16560510e-01, 1.40647197e+02, 3.35300000e+01, 2.16560510e-01,
        3.07805696e+02, 3.48652000e+01, 8.96710558e-02, 5.93764689e-01])
    # fmt: on

    np.testing.assert_allclose(obsfit.parameters, expvals, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    test_fitting_m101()
