from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scenicplus_core.scenicplus_core import algorithms as spc_algorithms

dtype_to_arg_sort_func = {
    np.int8: spc_algorithms.arg_sort_1d_i8,
    np.int16: spc_algorithms.arg_sort_1d_i16,
    np.int32: spc_algorithms.arg_sort_1d_i32,
    np.int64: spc_algorithms.arg_sort_1d_i64,
    np.uint8: spc_algorithms.arg_sort_1d_u8,
    np.uint16: spc_algorithms.arg_sort_1d_u16,
    np.uint32: spc_algorithms.arg_sort_1d_u32,
    np.uint64: spc_algorithms.arg_sort_1d_u64,
    np.float32: spc_algorithms.arg_sort_1d_f32,
    np.float64: spc_algorithms.arg_sort_1d_f64,
}

dtype_to_sort_func = {
    np.int8: spc_algorithms.sort_1d_i8,
    np.int16: spc_algorithms.sort_1d_i16,
    np.int32: spc_algorithms.sort_1d_i32,
    np.int64: spc_algorithms.sort_1d_i64,
    np.uint8: spc_algorithms.sort_1d_u8,
    np.uint16: spc_algorithms.sort_1d_u16,
    np.uint32: spc_algorithms.sort_1d_u32,
    np.uint64: spc_algorithms.sort_1d_u64,
    np.float32: spc_algorithms.sort_1d_f32,
    np.float64: spc_algorithms.sort_1d_f64,
}


def arg_sort(
    arr: npt.NDArray[np.int8]
    | npt.NDArray[np.int16]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.int64]
    | npt.NDArray[np.uint8]
    | npt.NDArray[np.uint16]
    | npt.NDArray[np.uint32]
    | npt.NDArray[np.uint64]
    | npt.NDArray[np.float32]
    | npt.NDArray[np.float64],
    reverse: bool = False,
    method: str | None = None,
) -> npt.NDArray[np.uint64]:
    """
    Returns the indices that would sort an 1D array.

    Perform an indirect sort using the algorithm specified by the `method` keyword.
    It returns an array of indices of the same shape as `arr` that index data in
    sorted order.

    Parameters
    ----------
    arr
        1D continuous numpy array.
    reverse
        If `True`, sort in descending order. Default: False (ascending order).
    method
        Method to use for sorting: "standard", "radix", "fastest"
        (uses standard or radix sort depending on the size).
        Default: None (= "fastest").

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Not a numpy array.")
    if arr.ndim != 1:
        raise ValueError("Not a 1D numpy array.")

    if method:
        if method not in ("standard", "radix", "fastest"):
            raise ValueError(
                f'Unsupported method: "{method}". Choose from "standard", "radix" or "fastest" (None).'
            )
    else:
        method = "fastest"

    arg_sort_func = dtype_to_arg_sort_func.get(arr.dtype.type)

    if arg_sort_func:
        return arg_sort_func(np.ascontiguousarray(arr), reverse, method)

    raise ValueError(
        f'Unsupported dtype "{arr.dtype}". Only np.int8-64, np.uint8-64 and np.float32-64 are supported.'
    )


def sort(
    arr: npt.NDArray[np.int8]
    | npt.NDArray[np.int16]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.int64]
    | npt.NDArray[np.uint8]
    | npt.NDArray[np.uint16]
    | npt.NDArray[np.uint32]
    | npt.NDArray[np.uint64]
    | npt.NDArray[np.float32]
    | npt.NDArray[np.float64],
    reverse: bool = False,
) -> npt.NDArray:
    """
    Sort an 1D array.

    Parameters
    ----------
    arr
        1D continuous numpy array.
    reverse
        If `True`, sort in descending order. Default: False (ascending order).

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Not a numpy array.")
    if arr.ndim != 1:
        raise ValueError("Not a 1D numpy array.")

    sort_func = dtype_to_sort_func.get(arr.dtype.type)

    if sort_func:
        return sort_func(np.ascontiguousarray(arr), reverse)

    raise ValueError(
        f'Unsupported dtype "{arr.dtype}". Only np.int8-64, np.uint8-64 and np.float32-64 are supported.'
    )
