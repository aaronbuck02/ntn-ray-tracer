#!/usr/bin/python3.12
"""Run the saved campaign with its existing Python 3.12 dependencies.

The host default Python is 3.14; the prepared NumPy 2 pickles refer to
numpy._core, whose equivalent in the installed NumPy 1.26 is numpy.core.
This launcher preserves the original campaign inputs and installed packages.
"""
import importlib
import runpy
import sys

sys.path.insert(0, "/path/to/sionna-venv/lib/python3.12/site-packages")
for module in ("numeric", "multiarray", "umath"):
    sys.modules[f"numpy._core.{module}"] = importlib.import_module(f"numpy.core.{module}")
sys.modules["numpy._core"] = importlib.import_module("numpy.core")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
