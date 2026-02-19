import numpy as np 
import pandas as pd 
import random
from typing import Literal


class HotDeckImputation:
    def __init__(self,df:pd.DataFrame, tickers:list[str] = [], n:int|None = None, **kwargs):
        self.df = df
        self.tickers = tickers 
        self.n = len(tickers) if not n else n

        #assert tickers or n, "At least a list of ticker symbols or the size of the portfolio is required!"
        assert self.n <= self.df.shape[1], f"Can't construct a n={self.n}-dimensional portfolio out of {self.df.shape[1]} assets!"


    def _best_donor(self,*args,**kwargs):
        raise NotImplementedError


    def impute(self):
        # Initial state
        receivers = self.tickers[:self.n]
        donors = [t for t in self.df.columns if t not in receivers]

        while len(receivers) < self.n:
            i_rnd = random.randint(0,len(donors)-1)
            if isinstance(receivers,np.ndarray):
                receivers = receivers.tolist()
            receivers.append(donors.pop(i_rnd))

        # Temporary dataframe that remove all nan-only rows; these cant be imputed by another asset; e.g. first row
        # of nans due to the return calculation
        df_r = self.df.dropna(axis=0, how="all").loc[:,receivers]
        df_r.columns = [i for i,_ in enumerate(receivers)]

        # Keeps tack of the current portfolio at all times
        df_p = pd.DataFrame(index=df_r.index, columns=df_r.columns)
        df_p.loc[:,:] = np.nan #receivers # initial values

        for i,symbol in enumerate(receivers):
            #print("-"*20)
            #print("Current symbol:", symbol)
            #print("-"*20)

            # Initial portfolio symbols where data is available
            # indices where the current symbol returns are not nan
            idx_notnan = df_r.index[df_r.loc[:,i].notna()]
            df_p.loc[idx_notnan,i] = symbol

            # as long as there's a missing value in the series, and as long as there's donors left
            while (n_nans:=df_r.loc[:,i].isna().sum()) > 0 and (n_donors:=len(donors))>0:
            
                #print(f"{n_nans} NaNs left")
                #print(f"{n_donors} donors left")

                # indices where the current symbol returns are nan
                idx_nan = df_r.index[df_r.loc[:,i].isna()] 

                # Possible donors and coverage on nan days
                possible_donors = self.df.loc[idx_nan,donors].notna().sum()
                #print("Possible donors:\n", possible_donors)

                # If there's no more donor that has data on missing days, jump to the next receiver
                if possible_donors.max() == 0:
                    #print("No more substitution possible, jumping to next symbol!")
                    break

                # Calculate the metric and choose the best donor
                metric, best_donor = self._best_donor(possible_donors, symbol)

                #print("Best donor:", best_donor)
                #print("Metric:", metric)

                # Fill temporary dataframe with the values of the best donor
                df_r.loc[idx_nan,i] = self.df.loc[idx_nan,best_donor]

                # Fill portfolio dataframe
                hasdata = self.df.loc[idx_nan,best_donor].notna()
                idx_hasdata = hasdata.index[hasdata]
                df_p.loc[idx_hasdata,i] = best_donor

                # donor already used once, can't be used a second time
                donors.remove(best_donor)

        #df_r.loc[self.df.isna().all(axis=1).index[0],:] = np.nan
        #df_p.loc[self.df.isna().all(axis=1).index[0],:] = np.nan 
    
        return df_p.sort_index(), df_r.sort_index()


class MaximumCoverageImputation(HotDeckImputation):
    """
    Hot-deck missing data imputation strategy where a de-listed company is recursively replaced by 
    another company with the maximum coverage on missing data days. If two donor companies tie on the
    coverage days it's randomly decided which to use for imputation.
    Every ticker can only be used once for imputation.

    **Arguments**:
    - df: DataFrame of returns with missing data size (i,j)
    - tickers: a list of k tickers that won't be used for imputation, but are desired components of the portfolio.
        These will appear in full-length in the imputed dataframe.
    - n: The size of the portfolio. If smaller then len(tickers), the first n entries are fixed portfolio
        components, the remaining are used for imputation. If n is larger then len(tickers), a random symbol
        chosen from the remaining is added to the specified tickers.

    **Returns**: 
    A tuple of two DataFrames:
    - (i,n) DataFrame of portfolio composition
    - (i,n) DataFrame of returns w/ replacement
    """

    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)


    def _best_donor(self, possible_donors, *args, **kwargs):
        max_coverage = possible_donors.max()
        best_donor = np.random.choice(possible_donors[possible_donors==max_coverage].index,size=1)[0]

        return max_coverage, best_donor


class MaximumCorrelationImputation(HotDeckImputation):
    """
    Hot-deck missing data imputation strategy where a de-listed company is recursively replaced by 
    another company with the maximum correlation coefficent, e.g. Person Correlation.
    Every ticker can only be used once for imputation.

    Arguments:
    - df: DataFrame of returns with missing data size (i,j)
    - tickers: a list of k tickers that won't be used for imputation, but are desired components of the portfolio.
        These will appear in full-length in the imputed dataframe.
    - n: The size of the portfolio. If smaller then len(tickers), the first n entries are fixed portfolio
        components, the remaining are used for imputation. If n is larger then len(tickers), a random symbol
        chosen from the remaining is added to the specified tickers.

    Returns: A tuple of two DataFrames:
    - (i,n) DataFrame of portfolio composition
    - (i,n) DataFrame of returns w/ replacement
    """

    def __init__(self,*args,method:Literal["pearson","kendall","spearman"]="pearson",**kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    
    def _best_donor(self, possible_donors, symbol, *args, **kwargs):
        _donors = possible_donors.index
        corr = self.df.loc[:,[symbol,*_donors]].corr(method=self.method).loc[_donors,symbol]
        max_corr = corr.max()
        best_donor = corr.idxmax()

        return max_corr, best_donor


class RandomImputation(HotDeckImputation):
    def __init__(self, df: pd.DataFrame, tickers: list[str] = [], n: int | None = None, **kwargs):
        super().__init__(df, tickers, n, **kwargs)

    def _best_donor(self, possible_donors, *args, **kwargs):
        best_donor = np.random.choice(possible_donors.index,size=1)[0]
        return np.nan, best_donor


class MaximumSimilarityImputation(HotDeckImputation):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        #super().__init__(*args,**kwargs)

    def _best_donor(self, possible_donors, *args, **kwargs):
        # TODO
        ...