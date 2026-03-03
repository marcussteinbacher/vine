import numpy as np

def simulate_hist(window:np.ndarray,weights:np.ndarray|None=None,alpha:float=0.01)->tuple[float,float]:
    '''
    Returns the VaR as empirical quantile and the ES as arithmetic mean of portfolio returns below the {alpha}-level.
    First, the portfolio returns are calculated from the asset returns in {window}, then the {alpha}-quantile
    is estimated and finally the mean of portfolio returns below the {alpha}-quantile (ES) is returned.
    '''
    n = window.shape[1]
    if not weights:
        weights = np.array([1/n]*n)
    
    mu_pfs = np.sum(weights*window,axis=1)
    var = np.quantile(mu_pfs,alpha)
    es = np.mean(mu_pfs,where=mu_pfs<var)
    
    return var.astype(np.float32), es.astype(np.float32)