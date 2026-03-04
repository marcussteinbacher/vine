import pyvinecopulib as pvc
from tools.Transformations import antithetic_variates, ppf_transform
from models.VineCopula import expected_shortfall, value_at_risk, VineCopulaResult
import numpy as np


_RND_KWARGS = ["seeds","num_threads","n","qrng"]
_RISK_KWARGS = ["weights","alpha"]
_MARGIN_KWARGS = ["f0"] # for t-margins, exclude df from estimation and set ot to a fixed value


def simulate_vc(
    window:np.ndarray,
    controls:pvc.FitControlsVinecop,
    margin_dist:str,
    n_samples:int=100_000,
    alpha:float=0.01,
    **kwargs
    )->tuple[VineCopulaResult, float, float]:

    rnd_kwargs = {k:v for k,v in kwargs.items() if k in _RND_KWARGS}
    risk_kwargs = {k:v for k,v in kwargs.items() if k in _RISK_KWARGS}
    margin_kwargs = {k:v for k,v in kwargs.items() if k in _MARGIN_KWARGS}

    u = pvc.to_pseudo_obs(window)
    vine = pvc.Vinecop.from_data(u, controls=controls)

    del u 

    sample = antithetic_variates(vine.simulate(n_samples, **rnd_kwargs), method="1-u")
    retrans, margin_params = ppf_transform(sample, window, distribution=margin_dist, **margin_kwargs)

    del sample, margin_params

    var = value_at_risk(retrans, alpha=alpha, **risk_kwargs)
    es = expected_shortfall(retrans, alpha=alpha,**risk_kwargs)

    del retrans

    return VineCopulaResult(vine), var, es