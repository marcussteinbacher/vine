import numpy as np
from models.VarianceCovariance import mu_pf_expected, sigma_pf_expected
from scipy.stats import norm, t
from scipy.integrate import quad


def simulate_varcov(window:np.ndarray,weights:np.ndarray|None=None,alpha:float=0.01,distribution:str='Normal',nu:float=3.0)->tuple[float,float]:
    '''
    Returns the VaR (quantile) and the expected tail return (ES) below the {alpha} level.

    Parameters:
    - distribution: 'Normal' or 'Student'
    - nu: degrees of freedom for Student-t (default 3)
    '''
    mu = mu_pf_expected(window,weights=weights)
    sigma = sigma_pf_expected(window,weights=weights)

    if distribution == 'Normal':
        var = mu + norm.ppf(alpha) * sigma
        def integrand(x):
            return x * norm.pdf(x, loc=mu, scale=sigma)

    elif distribution == 'Student':
        if nu <= 2:
            raise ValueError("nu must be > 2 for a finite variance when using Student-t")
        # Match the Student-t scale parameter to the provided sigma (portfolio std dev):
        # If X ~ t(df=nu, loc=0, scale=s) then Var(X) = s^2 * nu/(nu-2).
        # We want Var(X) = sigma^2, hence s = sigma * sqrt((nu-2)/nu).
        s = sigma * np.sqrt((nu - 2.0) / nu)
        var = mu + s * t.ppf(alpha, df=nu)
        def integrand(x):
            return x * t.pdf(x, df=nu, loc=mu, scale=s)

    else:
        raise ValueError("Unsupported distribution. Choose 'Normal' or 'Student'.")

    es = 1.0 / alpha * quad(integrand, -np.inf, var)[0]

    return var, es