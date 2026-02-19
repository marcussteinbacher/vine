import numpy as np

def value_at_risk(window:np.ndarray,weights:np.ndarray|None=None,alpha:float=0.01)->float:
    '''
    Returns the empirical {alpha}-quantile of portfolio returns calculated from asset returns in {window}. 
    First, the portfolio returns are calculated from the asset returns in {window}, then the {alpha}-quantile 
    is calculated.
    Models a continuous quantile function (see Hyndman & Fan, 1996).
    '''
    
    n = window.shape[1]
    
    if not weights:
        weights = np.array([1/n]*n)
    
    # Calculate portfolio returns
    mu_pfs = np.sum(weights*window,axis=1)
    
    return np.quantile(mu_pfs,alpha).astype(np.float32)


def expected_shortfall(window:np.ndarray,weights:np.ndarray|None=None,alpha:float=0.01)->float:
    '''
    Returns the arithmetic mean of portfolio returns below the {alpha}-level.
    First, the portfolio returns are calculated from the asset returns in {window}, then the {alpha}-quantile
    is estimated and finally the mean of portfolio returns below the {alpha}-quantile (ES) is returned.
    '''
    n = window.shape[1]
    if not weights:
        weights = np.array([1/n]*n)
    
    mu_pfs = np.sum(weights*window,axis=1)
    var = np.quantile(mu_pfs,alpha)
    
    return np.mean(mu_pfs,where=mu_pfs<var).astype(np.float32)