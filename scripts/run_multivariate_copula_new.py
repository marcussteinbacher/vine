import os 
import argparse
import pandas as pd
import copulae as cp
from functools import partial
from models import AdjustedReturn, MultivariateCopula
import time
import json
import config
import logging
from tools.Helpers import Parameters, StoreDict, save_scalars, save_objects, save_params
from tools.Runner import Runner
from simulations.MultivariateCopula import simulate_mvc
import multiprocessing as mp

# ------- Setting environment variables ----------
# suppress BLAS threading — Cholesky and matmul are the bottleneck,
# but with n_workers=cpu_count() you want 1 thread per worker
#os.environ["OMP_NUM_THREADS"]      = "1"
#os.environ["MKL_NUM_THREADS"]      = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["NUMBA_NUM_THREADS"] = "1"
# -------------------------------------------------

# Force spawn
mp.set_start_method("spawn", force=True)

_SIM = "MultivariateCopula"
logging.basicConfig(filename='./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description=f"Calculate the {_SIM} risk forecasts (VaR/ES). For intense computations calculates batches of data concurrently and writes the the results into a temporary folder. Afterwords, collects all temporary files and saves them into the simulation folder (Must be created manually!).")

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

    # Calculation params
    parser.add_argument("--max_workers",type=int,required=False,default=os.cpu_count(),help="Set the maximum number of CPU cores to use. Default: None, use all cores as of os.cpu_count(). Only effective when running in parallel.")
    parser.add_argument("--from",type=int,dest='start_idx',required=False,default=0,help="Start the calculation from a specific window index for distributed computing. Default: 0.")
    parser.add_argument("--to",type=int,dest="stop_idx",required=False,default=None,help="Stop the calculation at a specific window index for distributed computing. Default: None, calculate until the end.")

    # Save frequency
    parser.add_argument("--save_freq",type=int,required=False,default=-1,help="Save the object (e.g. fitted copula object) to disk every save_frequency windows. Default: -1, no saving.")
    parser.add_argument("--keep",action="store_true",required=False,help="Wether to keep the temporary files. Default: False.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Collecting all arguments into params
    _params = {"portfolio":args.portfolio,
              "volatility": {
                  "mean_model":"ConstantMean",
                  "volatility_model":args.volatility_model,
                  "innovation_distribution":args.innovation_distribution
                },
                "simulation": {
                    "name":_SIM,
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
                    "from":args.start_idx,
                    "to":args.stop_idx,
                    "save_freq":args.save_freq,
                    "parallel":True,
                    "max_workers":args.max_workers,
                    "keep":args.keep
                },
                "controls":args.controls
            }

    params = Parameters(_params)

    print(f"Starting {_SIM} VaR & ES with parameters:")
    print(params)

    logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{_SIM} with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)

    # Saving parameters to temp to be used to collect and saev DataFrames and to automatically resume calculation (not implemented yet)
    with open("temp/params.json","wt") as f:
        json.dump(params.dict,f,indent=4)

    print("Importing data...")
    returns = pd.read_parquet(f"data/{params.portfolio}/portfolio_returns.parquet")
    volatilities = pd.read_parquet(f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/volatility_forecasts.parquet")
    print("Done!")

    print("Creating adjusted return windows...")
    _, windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
    print("Done!")

    # Limit windows to --from, --to for distributed computing
    windows = windows[args.start_idx:args.stop_idx]
    fut_index = (returns.index + pd.offsets.BusinessDay(1))[args.start_idx:args.stop_idx] 

    del returns, volatilities

    # Set copula and margin distribution
    match args.copula:
        case "Gaussian":
            copula_cls = cp.GaussianCopula
        case "Student":
            # Little overhead high-speed implementation for 'itau'
            if args.fit_method == "itau":
                print("High-speed alternative chosen!")
                copula_cls = MultivariateCopula.MultivariateStudent
            elif args.fit_method == "irho":
                # Not recommended using irho
                print("High-speed alternative chosen!")
                copula_cls = MultivariateCopula.MultivariateStudent 
            else:
                copula_cls = cp.StudentCopula # copulae
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


    # Build the callable
    calc_fn = partial(simulate_mvc, 
                    copula_cls=copula_cls,
                    margin_dist=args.margin_distribution,
                    n_samples=args.n,
                    method=args.fit_method,
                    alpha=args.alpha,
                    **args.controls
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