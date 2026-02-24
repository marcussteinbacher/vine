import numpy as np
from copulae.elliptical.abstract import BaseCopula
from ._Simulation import value_at_risk, expected_shortfall
from tools.Transformations import antithetic_variates, ppf_transform
import json
import copulae as cp
from numba import njit
from tools.KendallTau import correlation_matrix as itau_correlation_matrix, make_positive_semidefinite
import scipy.stats as stats
from copulae.stats import multivariate_t as mvt, t
from copulae.types import Numeric
#import types
#import pycop as pc
import pandas as pd
from scipy.linalg import cholesky


def log_random(self, n:Numeric, seed:int|None=None):
    """
    Overwrite the original random method with one that uses logcdf.
    """
    r = mvt.rvs(cov=self.sigma, df=self._df, size=n, random_state=seed)
    log_u = t.logcdf(r, self._df)
    return np.exp(log_u)

# Monkey patch: All future instances will use the new random method
#cp.StudentCopula.random = log_random


class EmptyCopula(BaseCopula):
    """
    Empty copula class for windows with nan values.
    """
    def __init__(self, dim, name):
        super().__init__(dim=dim, name=name)
    
    def cdf(self, x):
        return None 
    
    def pdf(self, x):
        return None
    
    def params(self):
        return None


class MultiVariateCopulaResult:
    def __init__(self,copula,margin_dist:str,margin_params:np.ndarray):
        self.copula = copula
        self.margin_dist = margin_dist
        self.margin_params = margin_params.astype(np.float32) #.tolist()

    def to_dict(self):
        match self.margin_dist:
            case "Normal":
                params = {"mu":self.margin_params[0],"sigma":self.margin_params[1]}
            case "StudentsT":
                params = {"df":self.margin_params[0],"mu":self.margin_params[1],"sigma":self.margin_params[2]}
            case "Pareto":
                params = {"alpha":self.margin_params[0],"mu":self.margin_params[1],"sigma":self.margin_params[2]}
            case "Empirical":
                params = None
            case _ as e:
                raise ValueError(f"Unknown margin distribution: {e}")

        result_dict = {"copula":self.copula.name,
                       "dimension":self.copula.dim,
                       "margins":self.margin_dist,
                       "margin_params":params
                       }
        return result_dict

    def to_json(self,**kwargs):
        return json.dumps(self.to_dict(),**kwargs)
    
    def __repr__(self) -> str:
        if isinstance(self.copula, cp.StudentCopula):
            df_str = f"df={self.copula.params.df},"
        elif isinstance(self.copula, MultivariateStudent):
            df_str = f"df={self.copula.df},"
        else:
            df_str = ""
        return f"{self.copula.name}(dim={self.copula.dim},{df_str}margins={self.margin_dist})"
    
    def to_frame(self):
        ...


INIT_KWARGS = ["df"]
FIT_KWARGS = ["method","to_pobs","fix_df"] # fix_df not needed for itau, irho
RND_KWARGS = ["n","seed"]
RISK_KWARGS = ["weights","alpha"]
MARGIN_KWARGS = ["f0"] # for t-margins, exclude df from estimation and set ot to a fixed value


def simulate(window:np.ndarray,copula_cls:type[BaseCopula],margin_dist:str,n_samples:int=100_000,**kwargs):
    """
    Simulates alpha-level value-at-risk and expected shortfall based on a multivariate copula random sample incl. antithetic variates.
    Transforms window into pseudo-observations, then fits a copula to the pseudo-observations and 
    draws a random sample from the fitted copula.
    The sample is appended with antithetic variates and then transformed back to the desired margins.
    Returns the copula information, the alpha-level quantile, and the expected value bexýond the alpha-level.
    
    **Arguments**:
    - window: window of log-returns
    - copula_cls: A copulae copula
    - n_samples: int, Number of random samples,
    - margin_dist: str, margin distribution used for re-scaling, {'Empirical', 'Normal', 'StudentsT','Pareto'}
    - kwargs:
        - init_kwargs: Apply to copula.__init__().
            - 'df': float, default=1, only for StudentCopula
        - fit_kwargs: Apply to copula.fit().
            - 'method': str, {'ml', 'irho', 'itau'}, default='ml' 
            - 'to_pobs': bool, default=True
            - 'fix_df': bool, default=False, only effective for a StudentCopula and method 'itau'
        - rnd_kwargs: Apply to copula.random().
            - 'seed': int, default=None 
        - margin_kwargs: Apply to scipy.stats.t
            - 'f0': float, exclude df from estimation and set it to a fixed value
        - risk_kwargs: Apply to value-at-risk and expected-shortfall estimation
            - 'weights': np.array, default=None
            - 'alpha': float, default=0.01
                
    **Returns**: A tuple of the fitted copula, value-at-risk, and expected-shortfall.
        - MultivariateCopulaResult
        - var: Value at risk
        - es: Expected shortfall
    """
    
    init_kwargs = {k:v for k,v in kwargs.items() if k in INIT_KWARGS}
    fit_kwargs = {k:v for k,v in kwargs.items() if k in FIT_KWARGS}
    rnd_kwargs = {k:v for k,v in kwargs.items() if k in RND_KWARGS}
    risk_kwargs = {k:v for k,v in kwargs.items() if k in RISK_KWARGS}
    margin_kwargs = {k:v for k,v in kwargs.items() if k in MARGIN_KWARGS}
    
    pobs = cp.pseudo_obs(window)

    if np.any(np.isnan(window)):
        # If the window contains np.nans, skip the copula calibration and return nan
        #logging.info("Window contains np.nan, skipping copula calibration!")
        cop_obj, var, es = EmptyCopula(dim=window.shape[1],name="Empty"), np.nan, np.nan
    
    else:
        if issubclass(copula_cls,cp.EmpiricalCopula):
            cop_obj = copula_cls(pobs, **init_kwargs) # Empirical copula doesnt need fitting
        else:
            cop_obj = copula_cls(dim=window.shape[1],**init_kwargs)
            
            # Overload default random method for each instance individually
            #cop_obj.random = types.MethodType(log_random,cop_obj)
            
            cop_obj.fit(pobs, **fit_kwargs)

        sample = cop_obj.random(n_samples,**rnd_kwargs)

        anti_sample = antithetic_variates(np.array(sample))
        
        rescaled_sample, margin_params = ppf_transform(anti_sample, window, margin_dist, **margin_kwargs)
        
        var = value_at_risk(rescaled_sample, **risk_kwargs)
        es = expected_shortfall(rescaled_sample, **risk_kwargs)
        
    return MultiVariateCopulaResult(cop_obj,margin_dist,margin_params), var, es


class MultivariateStudentJIT:
    """
    An alternative implementation that utilizes numba.
    """
    def __init__(self, dim:int, df:float=3):
        self.name = "Student"
        self.dim = dim
        self.df = df
        self._fitted = False
    
    def fit(self, pobs:np.ndarray, method:str="itau")->None:
        match method:
            case "itau":
                self.corr = itau_correlation_matrix(pobs)
            case _ as m:
                raise NotImplementedError(f"Method {m} not implemented!")

        self._fitted = True

    @staticmethod
    @njit(cache=True, fastmath=True)
    def __random(rho_psd, df, n_samples):
        """
        Numba-jitted multivariate Student-t generator.
        X = Z / sqrt(S/nu)
        """
        n_vars = rho_psd.shape[0]
        # Generate Correlated Normals (Cholesky decomposition)
        L = np.linalg.cholesky(rho_psd)
        Z_unweighted = np.random.standard_normal((n_samples, n_vars))
        Z = Z_unweighted @ L.T

        # Generate Chi-Square shocks
        S = np.random.chisquare(df, size=n_samples).reshape(-1, 1)

        # Create the Student-t samples
        X = Z / np.sqrt(S / df)
        return X

    def random(self, n:int=100_000):
        if not self._fitted:
            raise ValueError("Copula not fitted!")
        
        random_sample = self.__random(self.corr, self.df, n)

        log_u = stats.t.logcdf(random_sample, self.df)
    
        return np.exp(log_u)
    
class MultivariateStudent:
    """
    An alternative high-performance implementation for sampling a Student copula using `itau`.
    """
    def __init__(self, dim:int, df:float=3):
        self.name = "Student"
        self.dim = dim
        self.df = df
        self._fitted = False
    
    def fit(self, pobs:np.ndarray, method:str="itau")->None:
        match method:
            case "itau":
                tau_matrix = pd.DataFrame(pobs).corr(method="kendall").to_numpy()
            case _ as m:
                raise NotImplementedError(f"Method {m} not implemented!")

        rho_matrix = np.sin((np.pi/2)*tau_matrix)
        self._fitted = True
        
        self.corr = make_positive_semidefinite(rho_matrix)

    @classmethod
    def _sample(cls, n, m, corr_matrix, nu):
        v = [np.random.normal(0, 1, m) for i in range(0, n)]

        # Compute the lower triangular Cholesky factorization of rho:
        L = cholesky(corr_matrix, lower=True)
        z = np.dot(L, v)

        # generate a random variable r, following a chi2-distribution with nu degrees of freedom
        r = np.random.chisquare(df=nu,size=m)

        y = np.sqrt(nu/ r)*z
        u = t.logcdf(y, df=nu, loc=0, scale=1)

        return np.exp(u)
    

    def random(self, n:int=100_000):
        if not self._fitted:
            raise ValueError("Copula not fitted!")
        return self._sample(self.dim, n, self.corr, self.df)