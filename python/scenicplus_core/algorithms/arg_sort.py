import numpy as np

from scenicplus_core.scenicplus_core import algorithms as spc_algorithms

dtype_to_arg_sort_func = {
    np.int8: spc_algorithms.arg_sort_i8,
    np.int16: spc_algorithms.arg_sort_i16,
    np.int32: spc_algorithms.arg_sort_i32,
    np.int64: spc_algorithms.arg_sort_i64,
    np.uint8: spc_algorithms.arg_sort_u8,
    np.uint16: spc_algorithms.arg_sort_u16,
    np.uint32: spc_algorithms.arg_sort_u32,
    np.uint64: spc_algorithms.arg_sort_u64,
    np.float32: spc_algorithms.arg_sort_f32,
    np.float64: spc_algorithms.arg_sort_f64,
}


def arg_sort(arr: np.ndarray, method: str | None) -> np.ndarray:
    """
    Returns the indices that would sort this 1D array.

    Parameters
    ----------
    arr
        1D continous numpy array.
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
        return arg_sort_func(arr, method)

    raise ValueError(
        f'Unsupported dtype "{arr.dtype}". Only np.int8-64, np.uint8-64 and np.float32-64 are supported.'
    )
