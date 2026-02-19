import numpy as np 
import pandas as pd
from tqdm import tqdm


class DefaultSlicer:
    """
    Provides window slicing/de-slicing functionality by slicing a (n,m)-DataFrame into 1-step 
    equally sized windows of shape (size, m). 
    """
    @staticmethod
    def sliding_window_view(df:pd.DataFrame, size=250)->tuple[np.ndarray,np.ndarray]:
        """
        Returns the default sliding window view with 250 observations in each window.
        Returns a tuple of (windowed index, window).
        """
        window_size = (size, df.shape[1]) # rows, cols
        windows = np.lib.stride_tricks.sliding_window_view(df, window_size).squeeze(1)
        windows_indices = np.lib.stride_tricks.sliding_window_view(df.index, window_size[0])
        return windows_indices, windows

    @staticmethod
    def reverse_sliding_window_view(windows:np.ndarray, index=None, columns=None)->pd.DataFrame:
        """
        Unrolls a windowed array into a DataFrame always keeping the first item in an window and filling towards the end with the remaining items in the last window. 
        Reverse operation of the default sliding window view operation based on np.lib.stride_tricks.sliding_window_view.
        Parameters:
        - index: Pandas index of the original DataFrame/Series before sliding_window
        - columns: Column names
        """
        rows = []
        for i,window in tqdm(enumerate(windows)):
            if i != len(windows)-1:
                rows.append(window[0])
            else: # for the last window
                rows.extend(window)
    
        return pd.DataFrame(
            np.array(rows),
            index=index,
            columns=columns
        )
 
    
def max_consecutive_windows(s:pd.Series,value:float=0.0)->pd.DataFrame:
    """
    Returns a DataFrame containing the windows of the maximum number consecutive values within a pandas Series. 
    A window is specified by a start-date, an end-date and the size of consecutive values. 

    |start|end|size|

    """
    _df = pd.DataFrame(s[s==value])
    
    _df['group'] = (s != s.shift()).cumsum()
    _groups = _df.groupby('group').count().where(lambda x: x > 1).dropna().astype(int)

    if len(_groups) == 0: # Theres no zero-return window longer than one day, return an empty DataFrame
        #return None
        return pd.DataFrame(columns=["start","end","size"])

    max_window_size = _groups.max().values[0]
    max_window_groups = _groups.loc[_groups[s.name]==max_window_size]

    windows = _df.loc[_df["group"].isin(max_window_groups.index.values)]
    windows.reset_index(inplace=True)
    windows = windows.loc[:,["date","group"]]

    starts = windows.groupby("group").min()
    ends = windows.groupby("group").max()

    return pd.DataFrame.from_dict({"start":starts.values.flatten(),"end":ends.values.flatten(),"size":max_window_size})