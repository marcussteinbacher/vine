import numpy as np

def make_positive_semidefinite(A:np.ndarray)->np.ndarray:
    # ensure positive semi-definite
    eigvals = np.linalg.eigvalsh(A)
    if eigvals.min() < 0:
        A += (-eigvals.min() + 1e-6) * np.eye(A.shape[0])
    return A