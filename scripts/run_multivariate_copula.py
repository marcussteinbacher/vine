import argparse
import pandas as pd
import numpy as np
import copulae as cp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from models import AdjustedReturn, MultivariateCopula
import time
import json
import config
import sys
import itertools
import logging
import pickle as pkl
from tools.Helpers import Parameters, chunks, StoreDict


SIM = "MultivariateCopula"

parser = argparse.ArgumentParser(description=f"Calculate the {SIM} risk forecasts (VaR/ES). For intense computations writes the batch-wise calculated risk forecasts in temp/portfolio. Afterwords, run `build_risk_dataframe.py` to aggregate the temporary files into a single DataFrame. Writes three objects, the fitted copula parameters in each window, the series of VaR forecasts, and the series of ES forecasts.")

parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the risk forecasts.")

# Volatility model params
parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility porcesse's innovation distribution.")

# Mutlivariate copula params
parser.add_argument("-cp","--copula",choices=config.MULTIVARIATECOPULAS,required=True,type=str,help="Set a multivariate copula the simulate VaR and ES.")
parser.add_argument("-md","--margin_distribution",choices=config.MARGINDISTRIBUTIONS,required=True,type=str,help="Set the margin distribution. Used to re-transform the uniformly distributed copula random samples.")
parser.add_argument("-n",type=int,required=False,default=100_000,help="Number of random samples drawn from copula to simulate VaR/ES. Default: 100_000")
parser.add_argument("-fm","--fit_method",type=str,required=False,default="ml",choices=["ml","itau","irho"],help="Choose the copula fitting method. Default: ml. Info: Ignored for EmpiricalCopula which doesn't need fitting.")

parser.add_argument("--controls",nargs='+',required=False,default={},action=StoreDict,help="Set additional copula parameters to be passed to MultivariateCopula.simulate, e.g. --controls df=3 df_fixed=True for a StudentCopula with fixed 3 degrees of freedom. Or, in case of StudentsT margins, f0=3 to exclude the degree of freedom parameter from being estimated in the re-transformation.")

# Risk metric params
parser.add_argument("-a","--alpha",required=False,type=float,default=0.01,help="Set alpha for the desired alpha-level VaR/ES, default 0.01 for the 1%%-VaR/ES.")

# Computation params
parser.add_argument("-cs","--chunk_size",type=int,required=False,default=64,help="Set the chunk-size for RAM-intense computations. Calculates chunks of chunk_size windows concurrently and writes the temporary results to disk, then continues with the next chunk_size windows. Default: 64.")

# Distributed computing
parser.add_argument("--from",type=int,dest='start',required=False,default=0,help="Start the calculation from a specific window index for distributed computing. Default: 0, start from the beginning, e.g. with the first window.")
parser.add_argument("--to",type=int,dest="stop",required=False,default=None,help="Stop the calculation at a specific window index for distributed computing. Default: None, calculate until the end, e.g until the last window is finished.")

# Save frequency
parser.add_argument("--save_freq",type=int,required=False,default=-1,help="Save the fitted multivariate copula object to disk every save_frequency windows. Default: -1, no saving.")

args = parser.parse_args()

args.simulation = SIM

# Collectin all arguments into params
_params = {"portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "simulation": {
                "name":args.simulation,
                "copula":args.copula,
                "margin_distribution":args.margin_distribution,
                "n":args.n,
                "fit_method":args.fit_method,
                "alpha":args.alpha,
                "risk_metric":[
                    "VaR",
                    "ES"
                ]
            },
            "calculation": {
                "chunk_size":args.chunk_size,
                "from":args.start,
                "to":args.stop,
                "save_freq":args.save_freq,
                "parallel":True
            },
            "controls":args.controls
        }

params = Parameters(_params)

print(f"Starting {SIM} {' & '.join(params.simulation.risk_metric)} with parameters:")
print(params)



logging.basicConfig(filename=f'./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{args.simulation} with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)


# Saving parameters to temp to be used to build volatility DataFrames and to automatically resume calculation (not implemented yet)
with open("temp/params.json","wt") as f:
    json.dump(params.dict,f,indent=4)

print("Importing data...")
returns = pd.read_parquet(f"data/{params.portfolio}/portfolio_returns.parquet")
volatilities = pd.read_parquet(f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/volatility_forecasts.parquet")
print("Done!")

print("Creating adjusted return windows...")
windows_indices, windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
print("Done!")

del returns, volatilities


# Restrict here to --from -> args.start, --to -> args.stop if calculation is distributed to different machines
windows_indices = windows_indices[args.start:args.stop]
windows = windows[args.start:args.stop]

# Start/Stop for distributed computing
start_chunk_idx = args.start//args.chunk_size


# Set copula and margin distribution
match args.copula:
    case "Gaussian":
        copula_cls = cp.GaussianCopula
    case "Student":
        copula_cls = cp.StudentCopula
    case "Empirical":
        copula_cls = cp.EmpiricalCopula
    case "Clayton":
        copula_cls = cp.ClaytonCopula
    case "Gumbel":
        copula_cls = cp.GumbelCopula
    case "Frank":
        copula_cls = cp.FrankCopula
    case _ as e:
        raise NotImplementedError(f"Copula {e} not implemented!")
    

print(f"Calculating {SIM} {' & '.join(params.simulation.risk_metric)}...")
start = time.perf_counter()

# -----------------
# START CALCULATION
# -----------------

for i, (chunk, chunk_idx) in tqdm(enumerate(zip(chunks(windows,args.chunk_size),chunks(windows_indices,args.chunk_size)),start=start_chunk_idx), total=len(windows)//args.chunk_size, desc="Chunks"):

    # Concurrent calculation 
    with ProcessPoolExecutor() as p:
        results = list(tqdm(p.map(partial(MultivariateCopula.simulate,copula_cls=copula_cls,margin_dist=args.margin_distribution,n_samples=args.n,method=args.fit_method,alpha=args.alpha,**args.controls),chunk), total=len(chunk), leave=False))
    
    # Transforming results
    cop_res, var, es = zip(*results)

    del results

    vars = np.array(var, dtype=np.float32)
    ess = np.array(es, dtype=np.float32)


    # Save copula in every n-th window
    win_idx = list(map(lambda x: x + i*args.chunk_size, range(args.chunk_size)))
    cops = filter(lambda t: t[0]%args.save_freq == 0, zip(win_idx,cop_res))

    del var, es, cop_res

    # Temporal alignment, set forecast one day into the future
    future_index = [idx[-1] + pd.offsets.BusinessDay(1) for idx in chunk_idx]

    var_series = pd.Series(vars, index=future_index)
    es_series = pd.Series(ess, index=future_index)

    del vars, ess

    # SAVE TEMPORARILY
    # Saving chunk-wise to save RAM
    var_series.to_frame(name="var").to_parquet(f"temp/var_{i:03d}.parquet")
    es_series.to_frame(name="es").to_parquet(f"temp/es_{i:03d}.parquet")
    # write if not empty
    if dict(cops):
        pkl.dump(dict(cops), open(f"temp/models_{i:03d}.pkl","wb"))

    del var_series, es_series, cops


end = time.perf_counter()

print(f"Calculation completed in {end-start:.2f} seconds!")
print(f"Run `build_risk_dataframes.py` to aggregate temporary results.")

logging.info(f"Calculation completed in {end-start:.2f} seconds!")