import numpy as np
import copulae as cp
from copulae.elliptical.abstract import BaseCopula
from models.MultivariateCopula import EmptyCopula, MultiVariateCopulaResult
from models._Simulation import value_at_risk, expected_shortfall
from tools.Transformations import antithetic_variates, ppf_transform


_INIT_KWARGS = ["df"]
_FIT_KWARGS = ["method","to_pobs","fix_df"] # fix_df not needed for itau, irho
_RND_KWARGS = ["n","seed"]
_RISK_KWARGS = ["weights","alpha"]
_MARGIN_KWARGS = ["f0"] # for t-margins, exclude df from estimation and set ot to a fixed value


def simulate_mvc(
    window:np.ndarray,
    copula_cls:type[BaseCopula],
    margin_dist:str,
    n_samples:int=100_000,
    **kwargs
    )->tuple[MultiVariateCopulaResult, float, float]:

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
    
    init_kwargs = {k:v for k,v in kwargs.items() if k in _INIT_KWARGS}
    fit_kwargs = {k:v for k,v in kwargs.items() if k in _FIT_KWARGS}
    rnd_kwargs = {k:v for k,v in kwargs.items() if k in _RND_KWARGS}
    risk_kwargs = {k:v for k,v in kwargs.items() if k in _RISK_KWARGS}
    margin_kwargs = {k:v for k,v in kwargs.items() if k in _MARGIN_KWARGS}
    
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