import numpy as np
import scipy.stats as stats
import pandas as pd

def antithetic_variates(sample:np.ndarray, method:str="1-u")->np.ndarray:
    """
    Extend a {sample} with antithetic variates for variance reduction without changing the law of the underlying simulation process (Glasserman et al., 2003, p.205).
    
    **Parameters**:
    sample: (n,m)-sized simulation
    method (optional): 
    - '1-u' for uniformly distributed samples (default)
    - '-z' for independent standard normal samples

    **Returns**: (2n, m)-sized ndarray of sample and appended antithetic variates
    """
    match method:
        case "1-u":
             res = np.append(sample,1-sample,axis=0)
        case "-z":
            res = np.append(sample,-sample,axis=0)
        case _:
            raise NotImplementedError(f"Method {method} not implemented!")
    return res


def ppf_transform(sample:np.ndarray, data:np.ndarray, distribution:str,**fit_kwargs)->tuple[np.ndarray,np.ndarray]:
    """
    Transforms a random {sample} drawn from a copula fitted with pseudo observations based on {data}
    back to its original distribution.

    **Arguments**:
    - sample: Sample drawn from a copula fitted with pseudo observations
    - data: original data
    - distribution: {'Normal','StudentsT','Pareto','Empirical'}
    - fit_kwargs: distribution fitting keyword arguments passed to scipy.stats.t|genpareto.fit

    **Returns**: 
    A tuple of:
    - trans: Sample transformed back to its original margin-distributions.
    - margin_params: Parameters of the original margin-distributions.
    """
    match distribution:
        case "norm"|"Normal":
            mu, sd = data.mean(axis=0), data.std(axis=0)
            trans = stats.norm.ppf(sample,loc=mu,scale=sd)
            margin_params = [mu,sd]
    
        case "t"|"StudentsT":
            df, mu, sd = [], [], []
            for i in range(data.shape[1]):
                params = stats.t.fit(data[:,i],**fit_kwargs)
                df.append(params[0])
                mu.append(params[1])
                sd.append(params[2])
            trans = stats.t.ppf(sample,df,loc=mu,scale=sd)
            margin_params = [df,mu,sd]

        case "pareto"|"Pareto":
            shape, mu, sd = [], [], []
            for i in range(data.shape[1]):
                params = stats.genpareto.fit(data[:,i],**fit_kwargs)
                shape.append(params[0])
                mu.append(params[1])
                sd.append(params[2])
            trans = stats.genpareto.ppf(sample,shape,loc=mu,scale=sd)
            margin_params = [shape,mu,sd]

        case "emp"|"Empirical":
            trans = np.array([np.quantile(col,q) for col,q in zip(data.T,sample.T)]).T
            margin_params = []

        case _:
            raise NotImplementedError(f"Not implemented for a {distribution} distribution!")
    
    return trans, np.array(margin_params,dtype=np.float32)

