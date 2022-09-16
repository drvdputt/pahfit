import numpy as np

from pahfit.helpers import read_spectrum
from pahfit.model import Model


def test_fitting_m101():

    # read in the spectrum (goes from 5.257 to 38.299)
    spectrumfile = "M101_Nucleus_irs.ipac"
    spec = read_spectrum(spectrumfile, spec1d=True)

    # Setup the model. Keep the old pack as a comment for later reference.
    # packfile = "scipack_ExGal_SpitzerIRSSLLL.ipac"
    packfile = "classic.yaml"
    # use a spitzer instrument model that covers the required range. SL1, SL2, LL1, LL2 should do
    instrumentname = "spitzer.irs.*.[12]"
    model = Model.from_yaml(packfile, instrumentname, 0)

    # fit
    model.guess(spec)
    model.fit(spec)

    # fmt: off
    expvals = np.array([
        6.58721942e-12, 5.00000000e+03, 1.90661338e-08, 3.00000000e+02,
        0.00000000e+00, 2.00000000e+02, 1.26333793e-06, 1.35000000e+02,
        3.51105886e-05, 9.00000000e+01, 1.36345070e-03, 6.50000000e+01,
        0.00000000e+00, 5.00000000e+01, 0.00000000e+00, 4.00000000e+01,
        1.95762968e-01, 3.50000000e+01, 0.00000000e+00, 5.27000000e+00,
        1.79180000e-01, 0.00000000e+00, 5.70000000e+00, 1.99500000e-01,
        3.10811829e+01, 6.22000000e+00, 1.86600000e-01, 3.72585550e+00,
        6.69000000e+00, 4.68300000e-01, 9.30119820e+00, 7.42000000e+00,
        9.34920000e-01, 3.26858957e+01, 7.60000000e+00, 3.34400000e-01,
        3.20674900e+01, 7.85000000e+00, 4.16050000e-01, 8.42546582e+00,
        8.33000000e+00, 4.16500000e-01, 2.48336838e+01, 8.61000000e+00,
        3.35790000e-01, 0.00000000e+00, 1.06800000e+01, 2.13600000e-01,
        2.63819267e+01, 1.12300000e+01, 1.34760000e-01, 3.28095146e+01,
        1.13300000e+01, 3.62560000e-01, 9.18251459e+00, 1.19900000e+01,
        5.39550000e-01, 2.04774690e+01, 1.26200000e+01, 5.30040000e-01,
        9.14042121e+00, 1.26900000e+01, 1.64970000e-01, 8.17467538e+00,
        1.34800000e+01, 5.39200000e-01, 1.49795930e+00, 1.40400000e+01,
        2.24640000e-01, 7.30127345e+00, 1.41900000e+01, 3.54750000e-01,
        0.00000000e+00, 1.59000000e+01, 3.18000000e-01, 1.58347389e+01,
        1.64500000e+01, 2.30300000e-01, 2.03825236e+01, 1.70400000e+01,
        1.10760000e+00, 8.79317728e+00, 1.73750000e+01, 2.08500000e-01,
        1.39420787e+00, 1.78700000e+01, 2.85920000e-01, 1.42928360e+00,
        1.89200000e+01, 3.59480000e-01, 2.63217124e+01, 3.31000000e+01,
        1.65500000e+00, 0.00000000e+00, 5.51356156e+00, 2.08185512e-02,
        0.00000000e+00, 6.10881078e+00, 2.02547771e-02, 2.12362756e+00,
        6.91319108e+00, 2.02547771e-02, 3.26303342e+00, 8.07580000e+00,
        4.67091295e-02, 0.00000000e+00, 9.67180375e+00, 4.67091295e-02,
        0.00000000e+00, 1.22875692e+01, 4.65865366e-02, 1.24259925e+01,
        1.70048925e+01, 5.35031847e-02, 1.25723210e+01, 2.81707000e+01,
        1.58811040e-01, 1.03933808e+01, 6.98871285e+00, 2.02547771e-02,
        0.00000000e+00, 8.98922850e+00, 4.20908173e-02, 0.00000000e+00,
        1.05170387e+01, 4.67091295e-02, 3.56991869e+01, 1.28333445e+01,
        4.67091295e-02, 5.78804484e+00, 1.55285270e+01, 6.53927813e-02,
        2.93408072e+01, 1.87403569e+01, 6.53927813e-02, 0.00000000e+00,
        2.59600000e+01, 1.58811040e-01, 0.00000000e+00, 2.59390000e+01,
        1.58811040e-01, 1.72720465e+02, 3.35300000e+01, 1.58811040e-01,
        2.55063734e+02, 3.48652000e+01, 1.29936306e-01, 0.00000000e+00])
    # fmt: on

    np.testing.assert_allclose(model.astropy_result.parameters, expvals, rtol=1e-6, atol=1e-6)
