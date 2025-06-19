import numpy as np
import numpy.typing as npt

import scenicplus_core


def p_adjust_bh(
    ps: npt.NDArray[np.float32] | npt.NDArray[np.float64] | list[float],
) -> npt.NDArray[np.float64]:
    """
    Adjust p-values to control the false discovery rate with Benjamini-Hochberg.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.
    p-values are adjusted using the Benjamini-Hochberg [1]_ control procedure.

    Parameters
    ----------
    ps
        The p-values to adjust. Elements must be real numbers between 0 and 1.

    Returns
    -------
    ps_adjusted
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    scipy.stats.false_discovery_control

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    """
    ps = np.asfarray(ps)
    by_descend = scenicplus_core.algorithms.arg_sort(ps, reverse=True)
    by_orig = scenicplus_core.algorithms.arg_sort(by_descend)
    steps = float(len(ps)) / np.arange(len(ps), 0, -1, dtype=np.float64)
    ps_adjusted = np.minimum(1, np.minimum.accumulate(steps * ps[by_descend]))
    return ps_adjusted[by_orig]
