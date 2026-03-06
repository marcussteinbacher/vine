import numpy as np 
import pandas as pd
from tools.Windows import DefaultSlicer
from models.Volatility import Egarch, Garch, GJR
from concurrent.futures import ProcessPoolExecutor
import arch.univariate as arch
from tqdm import tqdm
import logging
import time
import warnings
import argparse
import json
import config
from tools.Helpers import chunks, StoreDict
import pickle as pkl
import os
from tools.Helpers import save_checkpoint, get_checkpoint


parser = argparse.ArgumentParser(description="Calculate the volatility forecasts. Writes the batch-wise calculated volatility forecasts in temp. Afterwords, run `build_volatility_dataframe.py` to aggregate the temporary files into a single DataFrame.")

parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the volatility forecasts. Results will be stored in data/{portfolio}/...")

# Volatility model params
parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility porcesse's innovation distribution.")

#parser.add_argument("--hide_warning",action="store_true",help="Wether to hide scipy's convergence warnings while fitting the models. Changes the solvers `show_warning` parameter. Default: False, shows all warnings.")
#parser.add_argument("--rescale",action="store_false",help="Wether to scale the log-returns before fitting to improve convergence. Changes the models fitting parameter `rescale`. Default: True.")
parser.add_argument("--cov_type",choices=["robust","classic"],type=str,default="robust",required=False,help="This parameter specifies the method used to estimate the parameter covariance matrix during model fitting. Default: robust, `robust` applies the Bollersev-Wooldridge covariance estimator, `classic` corrsponds to the standard maximum likelihood covariance estimator.")
parser.add_argument("--tol",type=int,required=False,default=1e-6,help="Adjust the solver tolerance of `scipy.minimize` SQLSP solver, default 1e-6.")
parser.add_argument("--controls",nargs="+",required=False,default={}, action=StoreDict, help="Adjust further solver options. Valid entries include 'rescale','ftol', 'eps', 'disp', and 'maxiter'. Example: --controls maxiter=1000 (default). For possible options check scipy SLSQP.")

# Concurrent calculation
#parser.add_argument("--sequential", action="store_true",help="Wether to use sequential computation. Default: False, uses concurrent computation.")

# Calculation params
parser.add_argument("-bs","--batch_size",type=int,required=False,default=64,help="Set the batch-size for RAM-intense computations or interrupted cloud computing. Calculates batch-size windows concurrently and writes the temporary results to disk, then continues with the next batch of windows. Default: 64.")
parser.add_argument("--max_workers",type=int,required=False,default=None,help="Set the maximum number of CPU cores to use. Default: None, use all cores as of os.cpu_count(). Only effective when running in parallel.")
parser.add_argument("--from",type=int,dest='start',required=False,default=0,help="Start the calculation from a specific batch for distributed computing. Default: 0, start from the beginning, e.g. with the first batch.")
parser.add_argument("--to",type=int,dest="stop",required=False,default=None,help="Stop the calculation at a specific batch for distributed computing. Default: None, calculate until the end, e.g until the last batch is finished.")
parser.add_argument("--resume",action="store_true",required=False,help="Wether to resume the calculation from the last checkpoint, i.e. the last completed batch. Default: False.")

args = parser.parse_args()

if not args.max_workers:
    args.max_workers = os.cpu_count()

controls = dict(
    show_warning = True, # Always show convergence warnings
    rescale=True, # Always rescale to improve convergence
    tol=1e-6,
    maxiter=1000
)

controls.update(args.controls)

_params = {
          "portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "calculation": {
                "batch_size":args.batch_size,
                "from":args.start,
                "to":args.stop,
                "parallel": True, #not args.sequential,
                "max_workers":args.max_workers,
                "resume": args.resume,
            },
            "controls": controls 
        }


print("Starting volatility forecast with parameters:")
print(json.dumps(_params, indent=4))

# Saving parameters to temp to be used to build volatility DataFrames
with open("temp/params.json","wt") as f:
    json.dump(_params, f,indent=4)

#sys.exit()

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(filename='./log/volatility_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("\n"
             + "-" * 50
             + "\nStarting calculation with parameters:"
             + "\n" + json.dumps(_params, indent=2)
             + "\n" + "-" * 50)

# Load data
data = pd.read_parquet(f"data/{args.portfolio}/portfolio_returns.parquet")

print("Return data loaded!")

# Split data into windows
windows_indices, windows = DefaultSlicer.sliding_window_view(data, size=250)

# Calculate start, end batch indices
if args.resume:
    checkpoint = get_checkpoint()
    start, stop = checkpoint + 1, args.stop
else:
    start, stop = args.start, args.stop

index_batches = list(chunks(windows_indices, args.batch_size))[start:stop]
window_batches = list(chunks(windows, args.batch_size))[start:stop]

remaining_batches = len(window_batches)

del windows_indices, windows

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
        raise ValueError("Unknown combination of volatility model and innovation's distribution: {combo}!")


# Callback for parallel computation
def func(window):
    # Apply the volatility forecast column-wise to each window
    return np.apply_along_axis(model.volatility_forecast, axis=0, arr=window)

print(f"Caclulating volatility forecasts. Starting with batch {start}...")

# Calculate the volatility forecast for each chunk of windows and each window in the chunk concurrently
start_time = time.perf_counter()

for i, (batch, batch_idx) in tqdm(enumerate(zip(window_batches,index_batches),start=start), total=remaining_batches, desc="Batch"):
  
    # Concurrent calculation 
    with ProcessPoolExecutor(max_workers=args.max_workers) as p:
        results = list(tqdm(p.map(func,batch), total=len(batch),leave=False,desc="Window"))
    
    # Transforming results
    models, vols = zip(*results)

    vols = np.array(vols, dtype=np.float32)
    
    # Temporal alignment, set forecast one day into the future
    future_dates = [idx[-1] + pd.offsets.BusinessDay(1) for idx in batch_idx]

    df_vols = pd.DataFrame(vols, index=future_dates, columns=data.columns)

    df_models=pd.DataFrame(models, columns=data.columns, dtype=str)
    df_models.index.name = "window"

    # Calculating model statistics
    flat = df_models.map(lambda s: json.loads(s)["volatility"]).to_numpy().flatten()
    counts = {model:None for model in set(flat)}

    for key in counts.keys():
        counts[key] = int(np.count_nonzero(df_models.map(lambda s: json.loads(s)["volatility"]).to_numpy()==key))

    # Saving 
    with open(f"temp/volatility_models_summary_{i:03d}.pkl","wb") as f:
        pkl.dump(counts, f)

    # Saving chunk-wise to temp to save RAM
    df_vols.to_parquet(f"temp/volatility_forecasts_{i:03d}.parquet")
    #df_models.to_parquet(f"temp/volatility_models_{i:03d}.parquet")

    # Save progress 
    save_checkpoint(i)

    del results, models, vols, df_models, df_vols


end_time = time.perf_counter()

logging.info(f"Calculation completed in {end_time-start_time:.2f} seconds!")
logging.info(f"A total of {model.n.value} calculations and {model.nc.value} convergence failures!")

print(f"Calculation completed in {end_time-start_time:.2f} seconds!")
print(f"A total of {model.n.value} calculations and {model.nc.value} convergence failures!")
print("Run `build_volatility_data.py` to aggregate temporary results.")

# -----------
# ALTERNATIVE
# -----------

# Calculate all at once concurrently
#with ProcessPoolExecutor() as p:
#    results = list(tqdm(p.map(func,windows), total=len(windows)))

#models, vols = zip(*results)
#vols = np.array(vols, dtype=np.float32)

# Shifting each windows index + 1 day to align the one-day-ahead forecast
# future_dates = [index[-1] + pd.offsets.BusinessDay(1) for index in windows_indices]

# df_vols = pd.DataFrame(vols, index = future_dates, columns = data.columns)

# df_models = pd.DataFrame(models, columns=data.columns, dtype=str)
# df_models.index.name = "window"

# # Saving
# df_vols.to_parquet("data/{PORTFOLIO}/volatility_forecasts_{MODEL}_{DIST}.parquet")
# df_models.to_parquet("data/{PORTFOLIO}/volatility_models_{MODEL}_{DIST}.parquet")
