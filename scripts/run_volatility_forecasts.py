import numpy as np 
import pandas as pd
from tools.Windows import DefaultSlicer
from models.Volatility import Egarch, Garch, GJR
import arch.univariate as arch
import logging
import time
import warnings
import argparse
import json
import config
from tools.Helpers import StoreDict
from tools.Runner import Runner
import pickle as pkl
import os
from functools import partial
import multiprocessing as mp
from tools.Helpers import Parameters, save_vols, save_objects, save_params, _get_path

# Force spawn
mp.set_start_method("spawn", force=True)

logging.basicConfig(filename='./log/volatility_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the volatility forecasts. Writes the batch-wise calculated volatility forecasts in temp. Afterwords, run `build_volatility_dataframe.py` to aggregate the temporary files into a single DataFrame.")

    parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the volatility forecasts. Results will be stored in data/{portfolio}/...")

    # Volatility model params
    parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
    parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility porcesse's innovation distribution.")

    parser.add_argument("--cov_type",choices=["robust","classic"],type=str,default="robust",required=False,help="This parameter specifies the method used to estimate the parameter covariance matrix during model fitting. Default: robust, `robust` applies the Bollersev-Wooldridge covariance estimator, `classic` corrsponds to the standard maximum likelihood covariance estimator.")
    parser.add_argument("--tol",type=int,required=False,default=1e-6,help="Adjust the solver tolerance of `scipy.minimize` SQLSP solver, default 1e-6.")
    parser.add_argument("--controls",nargs="+",required=False,default={}, action=StoreDict, help="Adjust further solver options. Valid entries include 'rescale','ftol', 'eps', 'disp', and 'maxiter'. Example: --controls maxiter=1000 (default). For possible options check scipy SLSQP.")

    # Calculation params
    parser.add_argument("--max_workers",type=int,required=False,default=os.cpu_count(),help="Set the maximum number of CPU cores to use. Default: Use all cores as of os.cpu_count().")
    parser.add_argument("--from",type=int,dest='start_idx',required=False,default=0,help="Start the calculation from a specific window for distributed computing. Default: 0, start from the beginning, e.g. with the first window.")
    parser.add_argument("--to",type=int,dest="stop_idx",required=False,default=None,help="Stop the calculation at a specific window for distributed computing. Default: None, calculate until the end, e.g until the last window is finished.")
    parser.add_argument("--keep",action="store_true",required=False,default=False,help="Keep intermediate results. Default: False.")

    return parser.parse_args()


def volatility_window(window, model):
    # Apply the volatility forecast column-wise to each window, this runs in a loop.
    # Must be module level, so can't sit in main().
    return np.apply_along_axis(model.volatility_forecast, axis=0, arr=window)

def main():
    # Set some default controls
    controls = dict(
        show_warning = True, # Always show convergence warnings
        rescale=True, # Always rescale to improve convergence
        tol=1e-6,
        maxiter=1000
    )

    args = parse_args()

    controls.update(args.controls)

    # Setting params
    _params = {
          "portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "calculation": {
                "from":args.start_idx,
                "to":args.stop_idx,
                "parallel": True, #not args.sequential,
                "max_workers":args.max_workers,
            },
            "controls": controls 
        }

    # Collect parameters into Parameters object for chained dot-notation
    params = Parameters(_params)

    print("Starting volatility forecast with parameters:")
    print(params)

    # Saving parameters to temp to be used to build volatility DataFrames and to automatically resume calculation (not implemented yet)
    with open("temp/params.json","wt") as f:
        json.dump(params.dict,f,indent=4)


    logging.info("\n"
             + "-" * 50
             + "\nStarting calculation with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)

    # Load data
    returns = pd.read_parquet(f"data/{args.portfolio}/portfolio_returns.parquet")
    print("Return data loaded!")

    # Split data into windows
    windows_indices, windows = DefaultSlicer.sliding_window_view(returns)

    # Restrict data to --from, --to
    windows_indices = windows_indices[args.start_idx:args.stop_idx]

    fut_index = (returns.index + pd.offsets.BusinessDay(1))[args.start_idx:args.stop_idx]
    windows = windows[args.start_idx:args.stop_idx]


    # Initialize the model
    match (args.volatility_model, args.innovation_distribution):
        # GARCH
        case ("Garch", "Normal"):
            model = Garch(arch.ConstantMean, arch.Normal, **controls)
        case ("Garch", "StudentsT"):
            model = Garch(arch.ConstantMean, arch.StudentsT, **controls)
        case ("Garch", "SkewStudent"):
            model = Garch(arch.ConstantMean, arch.SkewStudent, **controls)
        case ("Garch", "GeneralizedError"):
            model = Garch(arch.ConstantMean, arch.GeneralizedError, **controls)
        case ("Garch", "Empirical"):
            model = Garch(arch.ConstantMean, arch.Normal,method="bootstrap", **controls)

        # EGARCH
        case ("Egarch", "Normal"):
            model = Egarch(arch.ConstantMean, arch.Normal, **controls)
        case ("Egarch", "StudentsT"):
            model = Egarch(arch.ConstantMean, arch.StudentsT, **controls)
        case ("Egarch", "SkewStudent"):
            model = Egarch(arch.ConstantMean, arch.SkewStudent, **controls)
        case ("Egarch", "GeneralizedError"):
            model = Egarch(arch.ConstantMean, arch.GeneralizedError, **controls)
        case ("Egarch", "Empirical"):
            model = Egarch(arch.ConstantMean, arch.Normal,method="bootstrap", **controls)

        # GJR-GARCH
        case ("GJR", "Normal"):
            model = GJR(arch.ConstantMean, arch.Normal,**controls)
        case ("GJR", "StudentsT"):
            model = GJR(arch.ConstantMean, arch.StudentsT,**controls)
        case ("GJR", "SkewStudent"):
            model = GJR(arch.ConstantMean, arch.SkewStudent,**controls)
        case ("GJR", "GeneralizedError"):
            model = GJR(arch.ConstantMean, arch.GeneralizedError,**controls)
        case ("GJR", "Empirical"):
            model = GJR(arch.ConstantMean, arch.Normal,method="bootstrap", **controls)

        case _ as combo:
            raise ValueError(f"Unknown combination of volatility model and innovation's distribution: {combo}!")
    
    # Build the callback for parallel computation
    calc_func = partial(volatility_window, model=model) # Needs to be pickable

    # Instantiate the runner
    runner = Runner(
        calc_func,
        data=windows,
        n_workers=args.max_workers,
        object_stride=1, # Save model in every window
        temp_dir="temp",
    )

    print("Caclulating volatility forecasts...")

    start_time = time.perf_counter()

    runner.run()

    end_time = time.perf_counter()

    print(f"Calculation completed in {end_time-start_time:.2f} seconds!")
    logging.info(f"Calculation completed in {end_time-start_time:.2f} seconds!")

    print(f"A total of {model.n.value} calculations and {model.nc.value} convergence failures!")
    logging.info(f"A total of {model.n.value} calculations and {model.nc.value} convergence failures!")

    # Collecting results
    print("Collecting results...")

    scalars = runner.collect_scalars()
    objects = runner.collect_objects()

    if not args.keep:
        runner.cleanup()

    # Saving results
    path = _get_path(params)

    index = fut_index[-len(scalars):]
    vol_matrix = np.stack([v[0] for v in scalars.values()]) # (N, n_assets), v[0] forecast, v[1] placeholder nan
    df_vol = pd.DataFrame(vol_matrix, index=index)
    df_vol.to_parquet(path+"volatility_forecasts.parquet")
    print(path+"volatility_forecasts.parquet written!")

    # Calculating model statistics - only save summary to save disk-space
    df_objects=pd.DataFrame.from_dict(objects, dtype=str, orient="index")
    df_objects.index.name = "window"

    flat = df_objects.map(lambda s: json.loads(s)["volatility"]).to_numpy().flatten()
    counts = {model:0 for model in set(flat)}

    for key in counts.keys():
        counts[key] = int(np.count_nonzero(df_objects.map(lambda s: json.loads(s)["volatility"]).to_numpy()==key))


    with open(f"{path}volatility_models_summary.json","wt") as f:
        json.dump(counts, f)
    
    print(path+"volatility_models_summary.json written!")

    save_params()
    print(path+"params.json written!")
    

if __name__ == "__main__":
    main()