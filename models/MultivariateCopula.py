import numpy as np
from copulae.elliptical.abstract import BaseCopula
import json
import copulae as cp
from tools.Kendall import kendall_tau_matrix, kendall_tau_matrix_nb
from tools.Spearman import spearman_rho_matrix
from tools.Matrices import make_positive_semidefinite
import scipy.stats as stats
from copulae.stats import multivariate_t as mvt, t
from copulae.types import Numeric
#import pandas as pd


def log_random(self, n:Numeric, seed:int|None=None):
    """
    Overwrite the original random method for t-copulas in the 'copulae'-package with one that uses logcdf.
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

    
class MultivariateStudent:
    """
    An alternative high-performance implementation for sampling a Student copula using `itau` or `irho`.
    Info: `itau` should be preferred over `irho` for copulas.
    """
    def __init__(self, dim:int, df:float=3):
        self.name = "Student"
        self.dim = dim
        self.df = df
        self._fitted = False
    
    def fit(self, pobs:np.ndarray, method:str="itau")->None:
        match method:
            case "itau":
                #tau_matrix = pd.DataFrame(pobs).corr(method="kendall").to_numpy()
                #tau_matrix = kendall_tau_matrix(pobs) # faster than pandas
                tau_matrix = kendall_tau_matrix_nb(pobs) # njit implememtation
                rho_matrix = np.sin((np.pi/2)*tau_matrix) 
            case "irho":
                spearman_rho = spearman_rho_matrix(pobs)
                rho_matrix = 2 * np.sin(np.pi / 6 * spearman_rho) # C-level call, faster
            case _ as m:
                raise NotImplementedError(f"Method {m} not implemented!")

        np.fill_diagonal(rho_matrix, 1.0)

        self._fitted = True
        
        self.corr = make_positive_semidefinite(rho_matrix)

    @classmethod
    def _sample(cls, corr_matrix, n_samples, dim, nu):
        # Cholesky + sample from t-copula
        
        L = np.linalg.cholesky(corr_matrix)
        df = nu
        z = np.random.standard_normal((n_samples, dim)) @ L.T
        chi2 = np.random.chisquare(df, size=(n_samples, 1))
        t_samples = z / np.sqrt(chi2 / df)

        # convert to uniform margins via t CDF
        u = stats.t.logcdf(t_samples,df=df)

        return np.exp(u)
    
    def random(self, n:int=100_000):
        if not self._fitted:
            raise ValueError("Copula not fitted!")
        return self._sample(self.corr, n, self.dim, self.df)