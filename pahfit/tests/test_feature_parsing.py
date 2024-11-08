from pahfit.model import Model
from pahfit.helpers import find_packfile
from pahfit.features import Features
from pahfit.errors import PAHFITModelError
import numpy as np


def test_feature_parsing():
    """

    Goal
    ----
    Test if the model is built successfully with certain features removed
    from the science pack. For example, when this test was written,
    generating the model without specifying any gaussians causes the
    code to crash. This test will try to provoke such crashes.

    This test does not check the correctness of the parsing, only the
    stability under certain edge cases. See the test_model_impl for
    something quantitative.

    Desired behavior
    ----------------

    The Fitter instance underlying model is generated correctly, without
    crashing.

    Functions that depend on specific model contents (lines, dust
    features, ...) can deal with those feature not being there.

    """
    # random instrument name
    instrumentname = "spitzer.irs.sl.2"

    # choose any science pack
    packfile = find_packfile("classic.yaml")

    # convert the yaml prescription to a features table
    features = Features.read(packfile)

    def test_parsing(features_edit):
        m = Model(features_edit)
        m._set_up_fitter(instrumentname, 0)
        m._ingest_fit_result_to_features()

    # Case 0: the whole table
    test_parsing(features)

    # Cases 1, 2, ...
    kinds = [
        "dust_continuum",
        "dust_feature",
        "line",
        "starlight",
        "attenuation",
    ]
    for kind in kinds:
        try:
            is_kind = features["kind"] == kind
            # anything but this kind
            test_parsing(features[np.logical_not(is_kind)])
            # only this kind
            test_parsing(features[is_kind])
            # only one feature of this kind?
            discard = is_kind  # discard everything of this kind
            discard[discard][0] = False  # except the first one
            test_parsing(features[np.logical_not(discard)])
        except PAHFITModelError:
            pass
            # if one of these is thrown, then the model sufficiently
            # warns about this edge case


if __name__ == "__main__":
    test_feature_parsing()
