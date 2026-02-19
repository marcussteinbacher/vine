import argparse
import pandas as pd
import numpy as np
import copulae as cp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from models import AdjustedReturn, VineCopula
from tools.Transformations import antithetic_variates, ppf_transform
import pyvinecopulib as pvc
import time
import json
import config
import sys
import itertools
import logging
import pickle as pkl
from tools.Helpers import Parameters, StoreDict, chunks


defaults = dict(
    parametric_method="itau", 
    family_set=[],
    selection_criterion="mbicv",
    #trunc_lvl = 7,
    preselect_families=True, # whether to exclude families based on symmetry of the data
    select_trunc_lvl=False,
    select_families=True, # select automatically if not given in family_set
    select_threshold=True, # automatically select threshold for thresholded vines
    num_threads=4
    )


SIM = "VineCopula"

parser = argparse.ArgumentParser(description=f"Calculate the {SIM} risk forecasts (VaR/ES). For intense computations writes the batch-wise calculated risk forecasts in temp/portfolio. Afterwords, run `build_risk_dataframe.py` to aggregate the temporary files into a single DataFrame. Writes three objects, the fitted copula parameters in each window, the series of VaR forecasts, and the series of ES forecasts.")

parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the risk forecasts.")

# Volatility model params
parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility model's innovation distribution.")

# Mutlivariate copula params
parser.add_argument("-cf","--copula_families",nargs='+',choices=config.VINECOPFAMILIES,required=False,default=[],type=str,help="Restrict the set of bivariate copulas that can be used to predict VaR/ES. Deault: Empty, all families can be used.")
parser.add_argument("-md","--margin_distribution",choices=config.MARGINDISTRIBUTIONS,required=True,type=str,help="Set the margin distribution. Used to re-transform the uniformly distributed copula random samples.")
parser.add_argument("-n",type=int,required=False,default=100_000,help="Number of random samples drawn from copula to simulate VaR/ES. Default: 100_000")
parser.add_argument("-fm","--fit_method",type=str,required=False,default="itau",choices=["ml","itau"],help="Choose the vine copula fitting method. Default: itau.")

# Vine controls
parser.add_argument("--controls",nargs="+",required=False,default={}, action=StoreDict,help="Overwrite the default vine copula controls. Possible entries w/ defaults include: parametric_method=itau, selection_criterion=mbicv, trunc_lvl=None, preselect_families=True, select_trunc_lvl=False, select_families=True, select_threshold=True, num_threads=4. Example: --controls trunc_lvl=7 preselect_families=True. Check pyvinecopulib.FitControlsVinecop for details.")

# Risk metric params
parser.add_argument("-a","--alpha",required=False,type=float,default=0.01,help="Set alpha for the desired alpha-level VaR/ES, default 0.01 for the 1%%-VaR/ES.")

# Computation params
parser.add_argument("-cs","--chunk_size",type=int,required=False,default=64,help="Set the chunk-size for RAM-intense computations. Calculates chunks of chunk_size windows concurrently and writes the temporary results to disk, then continues with the next chunk_size windows. Default: 64.")
parser.add_argument("--from",type=int,dest='start',required=False,default=0,help="Start the calculation from a specific window index for distributed computing. Default: 0, start from the beginning, e.g. with the first window.")
parser.add_argument("--to",type=int,dest="stop",required=False,default=None,help="Stop the calculation at a specific window index for distributed computing. Default: None, calculate until the end, e.g until the last window is finished.")

# Save frequency
parser.add_argument("--save_freq",type=int,required=False,default=-1,help="Save the vine copula object to disk every save_frequency windows. Default: -1, no saving.")

# Concurrent calculation
parser.add_argument("--sequential", action="store_true",help="Wether to use sequrntial computation. Default: False, uses concurrent computation.")

args = parser.parse_args()

args.fit_method = args.fit_method if args.fit_method == "itau" else "mle" # for API consistency: pyvinecopulib uses mle, copulae uses ml for maxmium-likelihood

# VineCopula controls
defaults.update(args.controls)
defaults["family_set"] = VineCopula.build_family_set(args.copula_families)
controls = pvc.FitControlsVinecop(**defaults)

# Collecting all arguments into params
_params = {"portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "simulation": {
                "name":SIM,
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
                "parallel": not args.sequential
            },
            "controls": VineCopula.get_controls(controls)
        }

params = Parameters(_params)

print(f"Starting {SIM} {' & '.join(params.simulation.risk_metric)} with parameters:")
print(params)

logging.basicConfig(filename=f'./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{SIM} with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)


# Saving parameters to temp to be used to build volatility DataFrames and to automatically resume calculation (not implemented yet)
with open("temp/params.json","wt") as f:
    json.dump(params.dict,f,indent=4)

print("Importing data...")
returns = pd.read_parquet(f"data/{args.portfolio}/portfolio_returns.parquet")
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


# Save current calculation params to temp to restore for building dataframes or resuming
with open("temp/params.json","wt") as f:
    json.dump(params.dict,f,indent=4)


def func(window):
    u = pvc.to_pseudo_obs(window)

    vine = pvc.Vinecop.from_data(u, controls=controls)

    del u 

    sample = antithetic_variates(vine.simulate(args.n), method="1-u")
    retrans, margin_params = ppf_transform(sample, window, distribution=args.margin_distribution)

    del sample, margin_params

    var = VineCopula.value_at_risk(retrans, alpha=args.alpha)
    es = VineCopula.expected_shortfall(retrans, alpha=args.alpha)

    del retrans

    return VineCopula.VineCopulaResult(vine), var, es


print(f"Calculating {SIM} VaR & ES...")
start = time.perf_counter()

# -----------------
# START CALCULATION
# -----------------

concurrent = not args.sequential

for i, (chunk, chunk_idx) in tqdm(enumerate(zip(chunks(windows,args.chunk_size),chunks(windows_indices,args.chunk_size)),start=start_chunk_idx), total=len(windows)//args.chunk_size, desc="Chunks"):

    if concurrent:
    # Concurrent calculation 
        with ProcessPoolExecutor() as p:
            results = list(tqdm(p.map(func,chunk), total=len(chunk), leave=False))
    else:
        results = []
    # Serial calculation
        for window in tqdm(chunk):
            results.append(func(window))
    

    # Transforming results
    vine_res, var, es = zip(*results)

    del results

    vars = np.array(var, dtype=np.float32)
    ess = np.array(es, dtype=np.float32)

    # Save vine in every n-th window
    win_idx = list(map(lambda x: x + i*args.chunk_size, range(args.chunk_size)))
    vines = filter(lambda t: t[0]%args.save_freq == 0, zip(win_idx,vine_res))

    del var, es, vine_res

    # Temporal alignment, set forecast one day into the future
    future_index = [idx[-1] + pd.offsets.BusinessDay(1) for idx in chunk_idx]

    var_series = pd.Series(vars, index=future_index)
    es_series = pd.Series(ess, index=future_index)

    del vars, ess

    # Saving chunk-wise to save RAM
    var_series.to_frame(name="var").to_parquet(f"temp/var_{i:03d}.parquet")
    es_series.to_frame(name="es").to_parquet(f"temp/es_{i:03d}.parquet")

    d = dict(vines)
    if d:
        print("IF",d)
        pkl.dump(d, open(f"temp/models_{i:03d}.pkl","wb"))

    del var_series, es_series, vines


end = time.perf_counter()

print(f"Calculation completed in {end-start:.2f} seconds!")
logging.info(f"Calculation completed in {end-start:.2f} seconds!")