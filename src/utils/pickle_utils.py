"""Load simulation pickles across the NumPy 1/2 core-module rename."""

import pickle

import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 writes array/scalar constructors under numpy._core; the
        # Sionna environment uses NumPy 1.26, where they live in numpy.core.
        # Remap only those constructors, without global sys.modules aliases.
        if np.__version__.split(".")[0] == "1" and module in {
            "numpy._core", "numpy._core.numeric", "numpy._core.multiarray",
            "numpy._core.umath",
        }:
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle(file):
    """Read a trusted simulation pickle from an open binary file.

    Saved numeric arrays and scalars retain their dtype, shape and values.
    This does not change installed NumPy or rewrite the prepared inputs.
    """
    return _NumpyCompatUnpickler(file).load()
