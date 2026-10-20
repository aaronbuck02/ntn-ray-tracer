"""Regression fixtures written by NumPy 2, read under NumPy 1 or 2."""

import base64
from io import BytesIO
import pickle

import numpy as np
import pytest

from Ray_Tracing.src.utils.pickle_utils import load_pickle


# NumPy 2.3.5: numeric arrays and scalars, using both reconstruction formats.
NUMPY2_PICKLES = [
    "gASV2gAAAAAAAAB9lCiMCXBvc2l0aW9uc5SMFm51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAxfcmVjb25zdHJ1Y3SUk5SMBW51bXB5lIwHbmRhcnJheZSTlEsAhZRDAWKUh5RSlChLAUsBSwOGlGgFjAVkdHlwZZSTlIwCZjiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAAAAAAAAPA/AAAAAAAAAEAAAAAAAAD4P5R0lGKMBnNjYWxhcpRoAowGc2NhbGFylJOUaBFDCAAAAAAAABxAlIaUUpR1Lg==",
    "gAWV3AAAAAAAAAB9lCiMCXBvc2l0aW9uc5SME251bXB5Ll9jb3JlLm51bWVyaWOUjAtfZnJvbWJ1ZmZlcpSTlCiWGAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAA+D+UjAVudW1weZSMBWR0eXBllJOUjAJmOJSJiIeUUpQoSwOMATyUTk5OSv////9K/////0sAdJRiSwFLA4aUjAFDlHSUUpSMBnNjYWxhcpSMFm51bXB5Ll9jb3JlLm11bHRpYXJyYXmUjAZzY2FsYXKUk5RoC0MIAAAAAAAAHECUhpRSlHUu",
]


@pytest.mark.parametrize("payload", NUMPY2_PICKLES)
def test_numpy2_array_and_scalar_preserved(payload):
    result = load_pickle(BytesIO(base64.b64decode(payload)))
    np.testing.assert_array_equal(result["positions"], [[1.0, 2.0, 1.5]])
    assert result["positions"].shape == (1, 3)
    assert result["positions"].dtype == np.dtype("float64")
    assert isinstance(result["scalar"], np.float64)
    assert result["scalar"] == 7


def test_current_numpy_and_metadata_round_trip():
    data = {"positions": np.array([[1, 2, 3]], dtype=np.int32), "meta": {"fixed_ue": True}}
    result = load_pickle(BytesIO(pickle.dumps(data)))
    np.testing.assert_array_equal(result["positions"], data["positions"])
    assert result["positions"].dtype == data["positions"].dtype
    assert result["meta"] == data["meta"]
