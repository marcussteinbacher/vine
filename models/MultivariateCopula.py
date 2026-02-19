import numpy as np
from copulae.elliptical.abstract import BaseCopula
from ._Simulation import value_at_risk, expected_shortfall
from tools.Transformations import antithetic_variates, ppf_transform
import json
import copulae as cp


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
    def __init__(self,copula:BaseCopula,margin_dist:str,margin_params:np.ndarray):
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
         else:
             df_str = ""
         return f"{self.copula.name}(dim={self.copula.dim},{df_str}margins={self.margin_dist})"
    
    def to_frame(self):
        ...


INIT_KWARGS = ["df"]
FIT_KWARGS = ["method","to_pobs","fix_df"]
RND_KWARGS = ["n","seed"]
RISK_KWARGS = ["weights","alpha"]
MARGIN_KWARGS = ["f0"] # for t-margins, exclude df from estimation and set ot to a fixed value


def simulate(window:np.ndarray,copula_cls:type[BaseCopula],margin_dist:str,n_samples:int=1_000_000,**kwargs):
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
            cop_obj.fit(pobs, **fit_kwargs)

        sample = cop_obj.random(n_samples,**rnd_kwargs)

        anti_sample = antithetic_variates(np.array(sample))
        
        rescaled_sample, margin_params = ppf_transform(anti_sample, window, margin_dist, **margin_kwargs)
        
        var = value_at_risk(rescaled_sample, **risk_kwargs)
        es = expected_shortfall(rescaled_sample, **risk_kwargs)
        
    return MultiVariateCopulaResult(cop_obj,margin_dist,margin_params), var, es

