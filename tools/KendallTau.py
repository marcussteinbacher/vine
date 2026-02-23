from numba import njit, prange
import numpy as np

"""
Kendall's tau helper functions for numba.
"""

@njit()
def update(bit, idx, val, n):
    """Update Binary Indexed Tree."""
    idx += 1
    while idx <= n:
        bit[idx] += val
        idx += idx & (-idx)


@njit()
def query(bit, idx):
    """Query prefix sum from Binary Indexed Tree."""
    idx += 1
    s = 0
    while idx > 0:
        s += bit[idx]
        idx -= idx & (-idx)
    return s


@njit()
def count_ties(arr):
    """Counts tie pairs sum(t*(t-1)/2) in a sorted array."""
    n = len(arr)
    ties = 0
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n and arr[i] == arr[j]:
            j += 1
        t = j - i
        if t > 1:
            ties += t * (t - 1) // 2
        i = j
    return ties


@njit(cache=True, fastmath=True)
def fast_tau_b(x, y):
    """Calculates Kendall's Tau-b in O(n log n)."""
    n = len(x)
    if n < 2: 
        return 1.0
    # Sort by X primarily, Y secondarily (Lexsort)
    perm = np.argsort(x)
    x_s, y_s = x[perm], y[perm]
    # Calculate tie counts
    n1 = count_ties(x_s)
    y_sorted_indep = np.sort(y)
    n2 = count_ties(y_sorted_indep)
    # Count joint ties (x and y tied simultaneously)
    n3 = 0
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n and x_s[i] == x_s[j] and y_s[i] == y_s[j]:
            j += 1
        t = j - i
        if t > 1: 
            n3 += t * (t - 1) // 2
        i = j
    # Map Y to ranks to use in Fenwick Tree
    y_ranks = np.searchsorted(y_sorted_indep, y_s)
    bit = np.zeros(n + 1, dtype=np.int64)
    concordant = 0
    for i in range(n):
        # query(ranks[i]-1) gets count of all strictly smaller ranks seen
        concordant += query(bit, y_ranks[i] - 1)
        update(bit, y_ranks[i], 1, n)
    n0 = n * (n - 1) // 2
    # Discordant formula adjusting for ties
    discordant = (n0 - n1 - n2 + n3) - concordant
    num = concordant - discordant
    den = np.sqrt(float(n0 - n1) * float(n0 - n2))
    return num / den if den != 0 else 0.0


@njit(parallel=True, cache=True)
def kendall_correlation_matrix(data):
    """Computes full Kendall Tau-b matrix in parallel."""
    n_obs, n_vars = data.shape
    matrix = np.eye(n_vars)
    for i in prange(n_vars):
        for j in range(i + 1, n_vars):
            tau = fast_tau_b(data[:, i], data[:, j])
            matrix[i, j] = tau
            matrix[j, i] = tau
    return matrix


@njit(cache=True,fastmath=True)
def fix_correlation_matrix(rho):
    """
    Ensures the matrix is Positive Semi-Definite using eigenvalue clipping.
    Efficiently jitted with Numba.
    """
    # eigh is supported by Numba (via LLVM/LAPACK)
    vals, vecs = np.linalg.eigh(rho)
    # Clip negative eigenvalues to a tiny positive floor
    vals = np.maximum(vals, 1e-9)
    # Reconstruct the matrix: V * diag(vals) * V^T
    psd_rho = vecs @ np.diag(vals) @ vecs.T
    # Re-normalize to ensure diagonal is exactly 1.0 (it's a correlation matrix)
    d = np.sqrt(np.diag(psd_rho))
    return psd_rho / np.outer(d, d)


def correlation_matrix(data):
    """
    Returns a PSD Kendall's Tau-b correlation matrix.
    """
    rho = kendall_correlation_matrix(data)
    return fix_correlation_matrix(rho)

def make_positive_semidefinite(rho):
    """
    Ensures the matrix is Positive Semi-Definite using eigenvalue clipping.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 1e-9)
    # Reconstruct the matrix: V * diag(vals) * V^T
    psd_rho = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to ensure diagonal is exactly 1.0 (it's a correlation matrix)
    d = np.sqrt(np.diag(psd_rho))
    return psd_rho / np.outer(d, d)