"""
Generation of fields and associated translation operators.

"""


import numpy as np
from .rusty_field import ffi, lib

__all__ = [
    "cube_sources"
]


def as_double_ptr(arr):
    """Turn to a double ptr."""
    return ffi.cast("double*", arr.ctypes.data)


def as_float_ptr(arr):
    """Turn to a float ptr."""
    return ffi.cast("float*", arr.ctypes.data)


def as_usize(num):
    """Cast number to usize."""
    return ffi.cast("unsigned long", num)


def as_double(num):
    """Cast number to double."""
    return ffi.cast("double", num)


def as_float(num):
    """Cast number to float."""
    return ffi.cat("float", num)


def align_data(arr, dtype=None):
    """Make sure that an array has the right properties."""

    if dtype is None:
        dtype = arr.dtype

    return np.require(arr, dtype=dtype, requirements=["C", "A"])


def cube_sources(p, origin=(0, 0, 0), length=1.0):
    """
    Create cube sources.
    """

    origin = np.array(origin, dtype=np.float64)

    result = np.empty((3, 6 * p * p), dtype=np.float64)

    lib.cube_sources_f64(
        as_usize(p),
        as_double(length),
        as_double_ptr(origin),
        as_double_ptr(result)
    )


    return result


