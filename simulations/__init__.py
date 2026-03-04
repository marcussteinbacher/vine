# simulations/__init__.py

"""
Force compilation of numba funcitons here. Once cached, every run uses the same.
"""
import numpy as np
from tools.Kendall import kendall_tau_matrix_nb
from tools.Transformations import empirical_ppf

def _warmup_numba():
    """
    Force compilation of numba functions here. Once cached, every run uses the same.
    """
    _d = np.random.rand(5, 5).astype(np.float32)
    _s = np.random.rand(5, 5).astype(np.float64)
    _sorted = np.sort(_d.astype(np.float64), axis=0)

    kendall_tau_matrix_nb(_d.astype(np.float32))
    empirical_ppf(_s, _sorted)

_warmup_numba()