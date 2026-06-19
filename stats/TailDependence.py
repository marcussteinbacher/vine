import numpy as np
from itertools import combinations

class Standard:
    """
    Standard conditional probability estimator (Sibuya 1960, Joe 1997):
    
    This requires choosing a threshold u and is sensitive to that choice — the classic bias-variance tradeoff.
    Info: u usually between 0.02 and 0.08.

    Usage:
    ```python
    u_grid = np.linspace(0.01, 0.10, 30)

    # From vine simulation
    vine_sim = fitted_vine.simulate(n=50_000)
    ltdc_sim = pairwise_ltdc(vine_sim, u_grid)

    # From in-sample PIT-transformed data (your empirical pseudo-obs)
    # pseudo_obs: (T, d) array of empirical uniform margins
    ltdc_data = pairwise_ltdc(pseudo_obs, u_grid)

    # Compare average across all pairs
    avg_sim  = average_ltdc(ltdc_sim,  u_grid)
    avg_data = average_ltdc(ltdc_data, u_grid)
    ```
    """
    @staticmethod
    def empirical_ltdc_pair(u_i: np.ndarray, u_j: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
        """
        Empirical lower tail dependence coefficient for a single pair.
        samples: (N, 2) array of uniform marginals
        u_grid: 1D array of threshold values in (0, 0.5)
        returns: array of lambda_L estimates at each u
        """
        return np.array([
            np.sum((u_i <= u) & (u_j <= u)) / np.sum(u_j <= u)
            for u in u_grid
        ])

    @staticmethod
    def pairwise_ltdc(samples: np.ndarray, u_grid: np.ndarray, 
                      pairs: list[tuple] | None = None) -> dict:
        """
        Compute empirical LTDC for all pairs (or a specified subset).
        samples: (N, d) uniform sample, e.g. draws from the vine
        returns: dict mapping (i, j) -> lambda_L array over u_grid
        """
        d = samples.shape[1]
        if pairs is None:
            pairs = list(combinations(range(d), 2))

        return {
            (i, j): Standard.empirical_ltdc_pair(samples[:, i], samples[:, j], u_grid)
            for i, j in pairs
        }

    @staticmethod
    def average_ltdc(ltdc_dict: dict) -> np.ndarray:
        """
        Portfolio-level summary: mean LTDC across all pairs at each threshold.
        Useful for comparing across different d.
        """
        return np.stack(list(ltdc_dict.values()), axis=0).mean(axis=0)
    

class CFG:
    """
    Frahm et al. (2005, p.10) CFG (= Caperaa-Fougeres-Genest, 1997)-inspired non-parametric estimator.

    The key difference is that it uses all observations rather than only those below 
    a threshold, and assumes the underlying copula can be approximated by an extreme 
    value copula.
    """

    @staticmethod
    def cfg_utdc_pair(u_i: np.ndarray, u_j: np.ndarray) -> float:
        """
        Frahm, Junker & Schmidt (2005) CFG estimator of the upper tail
        dependence coefficient for a single pair.

        u_i, u_j: 1D arrays of uniform marginals in (0, 1)

        Note: uses -log(u) transforms, so small u (lower tail) maps to
        large positive values, consistent with lower tail focus.
        """
        log_ui   = np.log(1.0 / u_i)                        # -log(u_i)
        log_uj   = np.log(1.0 / u_j)
        log_umax = 2.0 * np.log(1.0 / np.maximum(u_i, u_j)) # log(1/max^2) = 2*log(1/max)

        ratio = np.sqrt(log_ui * log_uj) / log_umax

        #return 2.0 - 2.0 * np.exp(np.mean(np.log(ratio)))

        # Frahm et al. (2005) suggest clipping to 0 for small negative values
        return max(0.0, 2.0 - 2.0 * np.exp(np.mean(np.log(ratio))))
    
    @staticmethod
    def cfg_ltdc_pair(u_i: np.ndarray, u_j: np.ndarray) -> float:
        """
        CFG estimator for LOWER tail dependence.
        Apply upper tail estimator to reflected sample (1-u_i, 1-u_j).
        """
        return CFG.cfg_utdc_pair(1.0 - u_i, 1.0 - u_j)


    @staticmethod
    def pairwise_ltdc(samples: np.ndarray,
                          pairs: list[tuple] | None = None) -> dict:
        """
        samples: (N, d) uniform draws from the vine
        returns: dict mapping (i, j) -> scalar CFG LTDC estimate
        """
        d = samples.shape[1]
        if pairs is None:
            pairs = list(combinations(range(d), 2))

        return {
            (i, j): CFG.cfg_ltdc_pair(samples[:, i], samples[:, j])
            for i, j in pairs
        }


    @staticmethod
    def average_ltdc(ltdc_dict: dict) -> float:
        return float(np.mean(list(ltdc_dict.values())))