import numpy as np 
import pandas as pd
from tools.Windows import DefaultSlicer

def adjusted_return_windows(returns:pd.DataFrame, volatilities:pd.DataFrame, n:int=250)->tuple[np.ndarray,np.ndarray]:
    """
    Returns the adjusted return window-indices and values for the given returns and volatility forecasts.
    Calculated in each window from *t* to *T* as:

    .. math::
    \\rho_t = \\frac{\\sigma_T}{\\sigma_t}

    First, the adjustment factors for each window are calculated such that the last adjustment
    factor in each window is one. Then, the returns are adjusted by the corresponding factors.

    **Parameters**:
    - returns: DataFrame (d x a) of returns, d days, a assets
    - volatilities: (d x a ) Array of volatility forecasts, already shifted for one-day forecast

    **Returns**: 
    - Tuple of the window-indices and the adjusted return windows, each of length n
    """
    # Common index 
    index = returns.index.intersection(volatilities.index)

    # Windowing
    window_indices, sigma_windows  = DefaultSlicer.sliding_window_view(volatilities.loc[index], n)
    _, return_windows = DefaultSlicer.sliding_window_view(returns.loc[index], n)

    assert sigma_windows.shape == return_windows.shape, "Shape mismatch between returns and volatility forecasts!"

    # Adjustment factors
    sigma_T = sigma_windows[:,-1,:] 
    sigma_T = sigma_T[:,np.newaxis,:] # boradcasting for efficient calculation

    adjustment_factor_windows = sigma_T / sigma_windows

    # Adjusted returns
    adjusted_return_windows = return_windows * adjustment_factor_windows

    return window_indices, adjusted_return_windows
