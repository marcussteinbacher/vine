import numpy as np
import multiprocessing
import arch.univariate as arch
import logging
import scipy.stats as stats 
from typing import Callable, Optional, Union
from collections.abc import Sequence
from arch._typing import ArrayLike, ArrayLike1D, Float64Array
from arch.univariate.base import ARCHModelResult
from scipy.integrate import quad
import json

class ARCHResult():
    """
    Wrapper class for arch.univariate.ARCHModelResult to provide a custom __repr__ method.
    Access the original result via the 'result' attribute.
    """
    def __init__(self, result:ARCHModelResult):
        self.result = result

    def to_json(self):
        """
        Returns a json string of the model's mean, volatility, innovation distribution and parameters to save as parquet.
        """
        mean_name = self.result.model.name
        vol_name = self.result.model.volatility.name
        dist_name = self.result.model.distribution.name

        if isinstance(self.result.model.volatility,(arch.GARCH, arch.EGARCH)):
            p,o,q = self.result.model.volatility.p, self.result.model.volatility.o, self.result.model.volatility.q
            desc = f"{vol_name}({p},{o},{q})"

        elif isinstance(self.result.model.volatility,arch.EWMAVariance):
            lam = self.result.model.volatility.lam
            desc = f"{vol_name.split("/")[0]}({lam})" # "EWMA/RiskMetrics"

        else:
            raise AttributeError("Unspecified volatility model! Implemented are GARCH, EGARCH or EWMAVariance")

        result_dict = {
            "mean":mean_name,
            "volatility":desc,
            "distribution":dist_name,
            "params":self.result.params.to_dict()
            }
        result_dict["params"] = {k:round(v,9) for k,v in result_dict["params"].items()} # round to save disk space
        return json.dumps(result_dict)
        
    def __repr__(self):
        return self.to_json()

 
class Empirical(arch.Distribution):
    """
    Empirical distribution to use as a distribution model for an ARCH/GARCH process in the arch package
    under the assumptions that the distribution of innovations mimics the empirical dstribution
    of the data.
    The empirical distribution of the data is estimated using a Gaussian kernel density estimate (KDE) and 
    a stats.rv_histogram of the provided data.
    """

    def __init__(
        self,
        resids: Float64Array,
        *,
        seed: Union[None, int, np.random.RandomState, np.random.Generator] = None,
    ) -> None:
        super().__init__(seed=seed)

        self._name = "Empirical"
        self.num_params: int = 0
        self._parameters: Optional[Float64Array] = None

        arr = np.asarray(resids, dtype=float)

        # Build KDE on standardized residuals (zero mean, unit std). The
        # log-likelihood evaluation during fitting will evaluate the KDE on
        # standardized residuals z = resids / sqrt(sigma2) and must account
        # for the Jacobian (divide by sqrt(sigma2)). Standardizing the
        # sample used to build the KDE makes the KDE represent the PDF of
        # standardized innovations.
        if arr.size == 0:
            arr = np.asarray([0.0])

        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if std == 0 or np.isnan(std):
            # constant data: add tiny jitter to create a usable bandwidth
            arr_std = (arr - mean) + np.random.normal(scale=1e-8, size=arr.shape)
            arr_std = arr_std / np.std(arr_std)
        else:
            arr_std = (arr - mean) / std

        self._resids = arr  # keep original raw values if needed

        self.counts, self.edges = np.histogram(arr_std, bins=20, density=True)

        #try:
        #    self._kde = stats.gaussian_kde(arr_std)
        #except (np.linalg.LinAlgError, ValueError):
        #    jitter = np.random.normal(scale=1e-8, size=arr_std.shape)
        #    self._kde = stats.gaussian_kde(arr_std + jitter)
    

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        return np.empty(0), np.empty(0)


    def bounds(self, resids: Float64Array) -> list[tuple[float, float]]:
        return []


    def loglikelihood(
        self,
        parameters: Union[Sequence[float], ArrayLike1D],
        resids: ArrayLike,
        sigma2: ArrayLike,
        individual: bool = False
        ) -> Union[float , Float64Array]:

        """
        Computes the log-likelihood of assuming residuals (innovations) follow the same 
        empirical distribution as the original data.

        Parameters
        ----------
        parameters : ndarray
            The empirical likelihood has no shape parameters. Empty since the
            empirical distribution has no shape parameters.
        resids  : ndarray
            The residuals to use in the log-likelihood calculation
        sigma2 : ndarray
            Conditional variances of resids
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is log(pdf(x)), where pdf is the kernel
        density estimate of the data.
        """

        # Evaluate density for standardized residuals z = resids / sqrt(sigma2).
        # The density of the (unstandardized) residual is pdf_z(z)/sqrt(sigma2).
        resids = np.asarray(resids, dtype=float)
        sigma2 = np.asarray(sigma2, dtype=float)

        # Ensure shapes align for elementwise operations
        sqrt_sigma2 = np.sqrt(sigma2)
        z = resids / sqrt_sigma2

        #pdf_z = self._kde.evaluate(z)

        #pdf_z = np.interp(z, self.edges[:-1],self.counts)

        #pdf_vals = pdf_z / sqrt_sigma2

        pdf_vals = np.interp(resids, self.edges[:-1],self.counts)

        # Avoid log(0) by replacing non-positive densities with a tiny positive value
        eps = np.finfo(float).tiny
        pdf_vals = np.where(pdf_vals <= 0, eps, pdf_vals)

        lls = np.log(pdf_vals)

        if individual:
            return np.asarray(lls)
        else:
            return float(np.sum(lls))


    def starting_values(self, std_resid: Float64Array) -> Float64Array:
        return np.empty(0)


    def _simulator(self, size: Union[int, tuple[int, ...]]) -> Float64Array:
        #return np.array(self._dist.rvs(size=size))
        return np.random.choice(self._resids, size=size)

    def simulate(
        self, parameters: Union[int, float, Sequence[Union[float, int]], ArrayLike1D]
    ) -> Callable[[Union[int, tuple[int, ...]]], Float64Array]:
        return self._simulator


    def parameter_names(self) -> list[str]:
        return []


    def cdf(
        self,
        resids: Union[Sequence[float], ArrayLike1D],
        parameters: Union[Sequence[float], ArrayLike1D, None] = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        
        #return self._dist.cdf(np.asarray(resids))
        x = np.sort(resids)
        y = np.arange(1, len(x) + 1) / len(x)
        return y


    def ppf(
        self,
        pits: Union[float, Sequence[float], ArrayLike1D],
        parameters: Union[Sequence[float], ArrayLike1D, None] = None,
    ) -> Float64Array:
        self._check_constraints(parameters)
        
        #ppf = self._dist.ppf(pits)
        ppf = np.quantile(self._resids,pits,dtype=np.float64)
        
        return ppf


    def moment(self, n:int, parameters:Union[Sequence[float], ArrayLike1D, None] = None) -> float:
        if n < 0:
            return np.nan
        
        return np.mean(self._resids**n)
        #return float(self._dist.moment(n))
        #return quad(lambda x: x**n * self._kde.pdf(self._resids),-np.inf,np.inf)
        

    def partial_moment(
        self,
        n: int,
        z: float = 0.0,
        parameters: Union[Sequence[float], ArrayLike1D, None] = None,
    ) -> float:
        
        r"""
        Order n lower partial moment from -inf to z

        Parameters
        ----------
        n : int
            Order of partial moment
        z : float, optional
            Upper bound for partial moment integral
        parameters : ndarray, optional
            Distribution parameters.  Use None for parameterless distributions.

        Returns
        -------
        float
            Partial moment

        References
        ----------
        .. [1] Winkler et al. (1972) "The Determination of Partial Moments"
               *Management Science* Vol. 19 No. 3

        Notes
        -----
        The order n lower partial moment to z is

        .. math::

            \int_{-\infty}^{z}x^{n}f(x)dx

        See [1]_ for more details.
        """

        def eval(x):
            return np.interp(x, self.edges[:-1],self.counts)

        if n < 0:
            return np.nan
        else:
            #return quad(lambda x: x**n *eval(x),-np.inf, z)[0] #quad(lambda x: x**n *self._kde.evaluate(x),-np.inf, z)[0]
            return quad(lambda x: x**n *self._kde.evaluate(x),-np.inf, z)[0]


class Garch:
    """
    Base class for the one-day-ahead volatility forecast. Model properties:
    - Mean model: specify an arch mean model, e.g. arch.univariate.ConstantMean
    - distribution model: specify an distribution model for the residuals, e.g. arch.univariatre.Normal, Voaltility.Empirial, etc.
    
    Volatility process resolution order in case of convergence issues: GARCH(1,1) -> EWMA(0.94)

    Returns: 
    Tuple of the fitted model and the one-day-ahead volatility forecast.
    """

    FITKWARGS = ["disp","tol","options","cov_type","show_warning","starting_values"]
    MEANKWARGS = ["rescale"]

    n = multiprocessing.Value('i', 0)  # Counts the total number of calculations
    nc = multiprocessing.Value('i', 0)  # Counts ConvergenceWarnings

    resolution_order = [arch.GARCH(p=1,o=0,q=1), arch.EWMAVariance(0.94)]

    def __init__(self, mean_model:type[arch.HARX], distribution_model:type[arch.Distribution],**kwargs:dict):
        """
        Initialize a Garch-model for one-day ahead volatiliy forecasting. Resolution order in case of
        convergence issues: Garch(1,1) -> EWMA(0.94).

        :param mean_model: Set the mean model, e.g. arch.univariate.ConstantMean
        :param distribution_model: Set the residuals distribution, e.g. arch.univariate.Normal, models.Volatility.Empirical
        :param kwargs: Additional keyword kwargs passed to: 

            - rescale:bool, wether to rescale data for better convergence.
            - cov_type:str, method used to estimate the parameter covariance matrix during model fitting. `robust` applies the Bollersev-Wooldridge covariance estimator, `classic` corrsponds to the standard maximum likelihood covariance estimator."
            - tol:float, adjust solver tolerance.
            - options:dict, a dictionary of further solver options, e.g. {"maxiter":1000}. See scipy.minimize for further information.
        """

        self.mean_model = mean_model
        self.distribution_model = distribution_model
        self.fit_kwargs = {k:v for k,v in kwargs.items() if k in self.FITKWARGS}
        self.mean_kwargs = {k:v for k,v in kwargs.items() if k in self.MEANKWARGS}

        #mean_model = arch.ConstantMean
        #distribution_model = Empirical # Distribution of the standardized residuals

    
    def volatility_forecast(self, series:np.ndarray)-> tuple[ARCHResult|float, float]:
        """
        One-day ahead volatility forecast for a series of returns. Returns a tuple of the fitted 
        model and the one-day-ahead volatility forecast.

        :param series: The series of returns.
        :type series: np.ndarray
        :return: A tuple of the model description with parameters, and the actual one-day ahead volatility forecast.
        :rtype: tuple[ARCHResult | float, float]
        
        """

        if np.isnan(series).any():
            # Can't calculate volatility forecast for a series that conatins nan
            return np.nan, np.nan
        
        am = self.mean_model(series,**self.mean_kwargs)

        if issubclass(self.distribution_model, Empirical):
            am.distribution = self.distribution_model(series)
        else:
            am.distribution = self.distribution_model()

        vol_forecast = np.nan
        
        for i,volatility_model in enumerate(self.resolution_order):
            with self.n.get_lock():
                self.n.value += 1

            am.volatility = volatility_model

            res = am.fit(disp=False,backcast=np.var(series)*0.94,**self.fit_kwargs) # lambda = 0.94 decay factor
            #res = am.fit(disp=False,**self.fit_kwargs)

            if not res.optimization_result.success:
                with self.nc.get_lock():
                    self.nc.value += 1
            
                logging.warning(f"No Convergence with {volatility_model.name}. Trying next model!")
                print(f"No Convergence with {volatility_model.name}. Trying next model!")

                continue
        
            else: # successful convergence
                if i>0:
                    print(f"Finally converged with {volatility_model.name}!")

                vol_forecast = res.forecast(horizon=1).variance.values[0][0]
                break 
        
        return ARCHResult(res), np.sqrt(vol_forecast/np.power(res.scale, 2)).astype(np.float32)
    
    @classmethod
    def nic(cls, epsilon, sigma_2, params):
        """
        Return the volatility answer to a shock event epsilon of a fitted
        GARCH model with initial variance sigma_2 and parameters params.
        Info: To compare the news impact curves of different models they must
        be fitted with rescale=False.
        """
        omega = params[0]
        alpha = params[1]
        beta = params[2]

        return omega + alpha*epsilon**2 + beta*sigma_2



class Egarch(Garch):
    """
    Base class for the one-day-ahead volatility forecast. Model properties:
    - Mean model: E.g. arch.Constant Mean
    - Distribution model: e.g. arch.Normal, arch.StudentsT, etc.

    Volatility process resolution order: EGARCH(1,1,1) -> GARCH(1,1) -> EWMA(0.94)
    """
    
    n = multiprocessing.Value('i', 0)  # Counts the total number of calculations
    nc = multiprocessing.Value('i', 0)  # Counts ConvergenceWarnings

    resolution_order = [arch.EGARCH(p=1,o=1,q=1), arch.GARCH(p=1,q=1), arch.EWMAVariance(0.94)]

    def __init__(self, mean_model, distribution_model,**kwargs):
        super().__init__(mean_model, distribution_model,**kwargs)

    @classmethod
    def nic(cls, epsilon, sigma_2, params):
        """
        Return the volatility answer to a shock event epsilon of a fitted
        EGARCH model with initial variance sigma_2 and parameters params.
        Info: To compare the news impact curves of different models they must
        be fitted with rescale=False.
        """
        omega = params[0]
        alpha = params[1]
        gamma = params[2] if len(params) == 4 else 0 
        beta = params[3] if len(params) == 4 else params[2]

        std = np.sqrt(sigma_2)
        e = epsilon/std 

        ln_sigma_2 = omega + alpha*(abs(e) - np.sqrt(2/np.pi)) + gamma*e + beta*np.log(sigma_2)

        return np.exp(ln_sigma_2)

    
class GJR(Garch):
    """
    Base class for the one-day-ahead volatility forecast. Model properties:
    - Mean model: E.g. arch.Constant Mean
    - Distribution model: e.g. arch.Normal, arch.StudentsT, etc.

    Volatility process resolution order: GJR-GARCH(1,1,1) -> GARCH(1,1) -> EWMA(0.94)
    """
    
    n = multiprocessing.Value('i', 0)  # Counts the total number of calculations
    nc = multiprocessing.Value('i', 0)  # Counts ConvergenceWarnings

    resolution_order = [arch.GARCH(p=1,o=1,q=1), arch.GARCH(p=1,q=1), arch.EWMAVariance(0.94)]

    def __init__(self, mean_model, distribution_model,**kwargs):
        super().__init__(mean_model, distribution_model,**kwargs)

    
    @classmethod
    def nic(cls, epsilon, sigma_2, params):
        """
        Return the volatility answer to a shock event epsilon of a fitted
        GJR-GARCH model with initial variance sigma_2 and parameters params.
        Info: To compare the news impact curves of different models they must
        be fitted with rescale=False.
        """
        omega = params[0]
        alpha = params[1]
        gamma = params[2]
        beta = params[3]

        def I(epsilon):
            return 1 if epsilon < 0 else 1

        return omega + alpha*epsilon**2 + gamma*epsilon**2 *I(epsilon) + beta*sigma_2