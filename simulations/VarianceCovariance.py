import numpy as np
from models.VarianceCovariance import mu_pf_expected, sigma_pf_expected
from scipy.stats import norm
from scipy.integrate import quad

def simulate_varcov(window:np.ndarray,weights:np.ndarray|None=None,alpha:float=0.01)->tuple[float,float]:
    '''
    Returns the VaR as analytical quantile and the expected tail return below the {alpha} level 
    quantile integrated from -inf to alpha.
    '''
    mu = mu_pf_expected(window,weights=weights)
    sigma = sigma_pf_expected(window,weights=weights)
    var = mu + norm.ppf(alpha)*sigma

    def integrand(x):
        return x*(1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2)))

    es =  1/alpha * quad(integrand,-np.inf,var)[0]
    #ES = mu - sigma*norm.pdf(norm.ppf(alpha))/(alpha)

    return var, es