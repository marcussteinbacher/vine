"""
simulations/copula.py

Serves as a template for simulations.

Copula fitting calculation. Each function receives a single (250, 50) window
and returns tuple[obj, float, float] as required by Runner.
"""

import warnings
import numpy as np

# ---------------------------------------------------------------------------
# Gaussian copula
# ---------------------------------------------------------------------------

def fit_gaussian_copula(
    window: np.ndarray,         # shape (250, 50)
    verbose: bool = False,
) -> tuple[object, float, float]:
    """
    Fit a Gaussian copula to a single window.

    Returns
    -------
    cop       : fitted GaussianCopula object
    log_lik   : scalar_a — log-likelihood of the fit
    aic       : scalar_b — Akaike information criterion
    """
    from copulae import GaussianCopula

    cop = GaussianCopula(dim=window.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cop.fit(window, verbose=verbose)

    log_lik = float(cop.log_lik(window))
    n_params = cop.params.size
    aic = float(-2 * log_lik + 2 * n_params)

    return cop, log_lik, aic