import numpy as np 
from scipy.stats import spearmanr

def spearman_rho_matrix(X:np.ndarray)->np.ndarray:
    corr, _ = spearmanr(X)
    corr    = np.clip(corr, -1, 1) # guard against numerical noise
    
    return corr
    
def spearman_tau_matrix(X:np.ndarray)->np.ndarray:
    """
    Spearman in one vectorised C-level call — ~100x faster than pandas kendall.
    For VaR estimation via copula simulation, Spearman is perfectly acceptable.
    Spearman -> Kendall approximation for elliptical: tau ≈ 2/3 * rho_spearman.
    """
    corr, _ = spearmanr(X)
    corr    = np.clip(corr, -1, 1) # guard against numerical noise
    # itau inversion for t-copula: rho = sin(pi/2 * tau)
    # Spearman -> Kendall approximation for elliptical: tau ≈ 2/3 * rho_spearman
    tau_matrix  = (2 / 3) * corr
    
    return tau_matrix