import pandas as pd
import numpy as np
import math as m

def mu(window:pd.DataFrame|np.ndarray,weights:np.ndarray|None=None)->pd.Series:
    """
    Calculates the portfolio returns for a DataFrame/window of single asset returns with individual {weights}.
    If weights are not specified equal weights are applied.
    """
    n = window.shape[1]

    #Default to equal weights if not specified
    if not weights:
        weights = np.array([1/n]*n)
        
    #Some sanity checks
    assert m.isclose(np.sum(weights),1)
    assert len(weights) == n

    if isinstance(window, pd.DataFrame):
        _mu = np.sum(weights*window.to_numpy(),axis=1)
        _index = window.index
    else:
        _mu = np.sum(weights*window,axis=1)
        _index = None

    return pd.Series(_mu, index=_index)


def std(window:np.ndarray,weights:np.ndarray|None=None)->pd.Series:
    """
    Calculate the portfolio standard deviation based on the last {len(window)} returns using a variance-covariance matrix.
    If no weights are specified assets are assumed to be equally weighted.
    """
    n = window.shape[1]

    #Default to equal weights if not specified
    if not weights:
        weights = np.array([1/n]*n)

    #Some sanity checks
    assert m.isclose(np.sum(weights),1)
    assert len(weights) == n

    cov = np.cov(window,rowvar=False)
    std = np.sqrt(weights.T.dot(cov).dot(weights))

    return std