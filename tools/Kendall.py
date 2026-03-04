import numba as nb
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations

"""
Kendall's tau helper functions.

`itau` is generally considered more robust for copulas than `irho`.
"""

def kendall_tau_matrix(X:np.ndarray)->np.ndarray:
    """
    A faster implementation of Kendall's tau matrix.
    Kendall's tau is a more natural concordance measure for copulas — it depends only on the copula 
    and not on the marginal distributions, whereas Spearman's rho has a small marginal-distribution 
    dependency.
    """
    n = X.shape[1]
    mat = np.eye(n)
    for i,j in combinations(range(n),2):
        tau,_ = kendalltau(X[:,i],X[:,j])
        mat[i,j] = mat[j,i] = tau
    return mat


@nb.njit(parallel=False, cache=True)
def _kendall_tau_pair(x, y):
    """Kendall tau for a single pair of variables."""
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sign = dx * dy
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
            # ties contribute 0
    return (concordant - discordant) / (0.5 * n * (n - 1))


@nb.njit(parallel=True, cache=True)
def kendall_tau_matrix_nb(X):
    """
    A faster numba implementation of Kendall's tau matrix.
    Full (dim, dim) Kendall tau matrix.
    X : (n_obs, dim) float32 — e.g. (250, 50)
    """
    n_obs, dim = X.shape
    mat = np.eye(dim, dtype=np.float64)

    for i in nb.prange(dim):            # parallel outer loop over columns
        for j in range(i + 1, dim):
            tau = _kendall_tau_pair(X[:, i], X[:, j])
            mat[i, j] = tau
            mat[j, i] = tau

    return mat