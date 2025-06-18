from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scenicplus_core.scenicplus_core import algorithms as spc_algorithms


def get_nonzero_row_indices(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint64]:
    """
    Get the indices of the rows that have at least one nonzero element.

    This function is equivalent to:
        np.nonzero(np.count_nonzero(x, axis=1))[0]

    Parameters
    ----------
    arr
        2D float32 continuous numpy array.

    Returns
    -------
    row indices of the rows that have at least one nonzero element.

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Not a numpy array.")
    if arr.ndim != 2:
        raise ValueError("Not a 2D numpy array.")
    if arr.dtype != np.float32:
        raise ValueError("Not a 2D float32 numpy array.")

    return spc_algorithms.get_nonzero_row_indices(np.ascontiguousarray(arr))
