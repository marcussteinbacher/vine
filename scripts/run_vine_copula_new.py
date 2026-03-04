import argparse
import pandas as pd
from models import AdjustedReturn, VineCopula
from simulations.VineCopula import simulate_vc
import pyvinecopulib as pvc
import time
import json
import config
import os
import logging
from tools.Helpers import Parameters, StoreDict, save_scalars, save_objects, save_params
from functools import partial
from tools.Runner import Runner


# ------- Setting environment variables ----------
# suppress BLAS threading — Cholesky and matmul are the bottleneck,
# but with n_workers=cpu_count() you want 1 thread per worker
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
# -------------------------------------------------


_SIM = "VineCopula"
logging.basicConfig(filename='./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Default controls for VineCopula
_defaults = dict(
    parametric_method="itau", 
    family_set=[],
    selection_criterion="mbicv",
    #trunc_lvl=7,
    preselect_families=True, # whether to exclude families based on symmetry of the data
    select_trunc_lvl=False,
    select_families=True, # select automatically if not given in family_set
    select_threshold=True, # automatically select threshold for thresholded vines
    num_threads=1 # Leave at 1 if using all cores
    )


def parse_args():
    parser = argparse.ArgumentParser(description=f"Calculate the {_SIM} risk forecasts (VaR/ES). Writes the calculated risk forecasts in temp/portfolio. Automatically resumes an aborted calculation. Saves the results into the simulation folder. The path must exist!")

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

    # Calculation params
    parser.add_argument("--max_workers",type=int,required=False,default=os.cpu_count(),help="Set the maximum number of worker processes. Default: All CPU cores, os.cpu_count(). Only effective when running in parallel.")
    parser.add_argument("--from",type=int,dest='start_idx',required=False,default=0,help="Start the calculation from a specific window index for distributed computing. Default: 0, start from the beginning, e.g. with the first window.")
    parser.add_argument("--to",type=int,dest="stop_idx",required=False,default=None,help="Stop the calculation at a specific window for distributed computing. Default: None, calculate until the end, e.g until the last window is finished.")

    # Save frequency
    parser.add_argument("--save_freq",type=int,required=False,default=-1,help="Save the vine copula object to disk every save_frequency windows. Default: -1, no saving.")
    parser.add_argument("--keep",action="store_true",required=False,help="Wether to keep the temporary files. Default: False.")

    return parser.parse_args()

def main():
    args = parse_args()
    args.fit_method = args.fit_method if args.fit_method == "itau" else "mle" # for API consistency: pyvinecopulib uses mle, copulae uses ml for maxmium-likelihood


    # VineCopula controls
    _defaults.update(args.controls)
    _defaults["family_set"] = VineCopula.build_family_set(args.copula_families)
    controls = pvc.FitControlsVinecop(**_defaults)


    # Collecting all arguments into params
    _params = {"portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "simulation": {
                "name":_SIM,
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
                "from":args.start_idx,
                "to":args.stop_idx,
                "save_freq":args.save_freq,
                "max_workers":args.max_workers,
            },
            "controls": VineCopula.get_controls(controls)
        }

    params = Parameters(_params)

    print(f"Starting {_SIM} VaR & ES with parameters:")
    print(params)

    logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{_SIM} with parameters:"
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

    # Limit windows to --from, --to for distributed computing
    windows = windows[args.start_idx:args.stop_idx]
    fut_index = (returns.index + pd.offsets.BusinessDay(1))[args.start_idx:args.stop_idx] 

    del returns, volatilities

    # Build the callable
    calc_fn = partial(simulate_vc, 
                    controls=controls,
                    margin_dist=args.margin_distribution,
                    n_samples=args.n,
                    alpha=args.alpha,
                    )

    # Instantiate the runner
    runner = Runner(
        calculation_fn=calc_fn,
        data=windows,
        n_workers=args.max_workers,
        max_in_flight=args.max_workers * 8,
        flush_threshold=16, # flush scalars every 16 completed windows
        object_stride=args.save_freq,
        object_flush_threshold=4, # flush objects when 4 are enqueued
        temp_dir="temp"
    )

    print(f"Calculating {_SIM} VaR & ES...")
    start_time = time.perf_counter()
    
    runner.run()

    end_time = time.perf_counter()

    print(f"Calculation completed in {end_time-start_time:.2f} seconds!")
    logging.info(f"Calculation completed in {end_time-start_time:.2f} seconds!")
    
    # Collecting results
    scalars = runner.collect_scalars()
    cops = runner.collect_objects()

    if not args.keep:
        runner.cleanup()

    # Save aggregated data: VaR.parquet, ES.parquet, models.pkl, params.json
    index = fut_index[-len(scalars):]
    save_scalars(scalars,index)
    save_objects(cops)
    save_params()
    

if __name__ == "__main__":
    main()