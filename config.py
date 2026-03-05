def set_env(n_threads=1):
    import os

    # set ALL of these before importing numpy, numba, scipy or any other library
    os.environ["OMP_NUM_THREADS"]        = f"{n_threads}"   # OpenMP — used by many C extensions
    os.environ["MKL_NUM_THREADS"]        = f"{n_threads}"   # Intel MKL (numpy/scipy if built against MKL)
    os.environ["OPENBLAS_NUM_THREADS"]   = f"{n_threads}"   # OpenBLAS (numpy/scipy if built against OpenBLAS)
    os.environ["BLIS_NUM_THREADS"]       = f"{n_threads}"   # BLIS (alternative BLAS, less common)
    os.environ["NUMBA_NUM_THREADS"]      = f"{n_threads}"   # numba prange / parallel njit
    os.environ["NUMEXPR_NUM_THREADS"]    = f"{n_threads}"   # numexpr (used by pandas internally)
    os.environ["PVC_NUM_THREADS"]        = f"{n_threads}"   # pyvinecopulib


# All created portfolios
PORTFOLIOS = [10, 20, 30, 50, 100, 200, 500, 1000, 1500]

# Volatility forecast settings
VOLATILITYMODELS = ["Garch","Egarch","GJR"]
INNOVATIONDISTRIBUTIONS = ["Normal","StudentsT","SkewStudent","GeneralizedError","Empirical"]

# Risk forecast resolution methods/simulations settings
SIMULATIONS = ["HistoricalSimulation","VarianceCovariance","MultivariateCopula","VineCopula"]
RISKMETRICS = ["VaR", "ES"]
MULTIVARIATECOPULAS = ["Gaussian","Student","Empirical","Clayton","Frank","Gumbel"]
MARGINDISTRIBUTIONS = ["Normal","StudentsT","Empirical","Pareto"]

VINECOPFAMILIES = ["Gaussian","Student","Clayton","Frank","Gumbel","Joe","Independent","BB1","BB6","BB7","BB8","Tawn","TLL"]