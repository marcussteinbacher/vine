import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from models import AdjustedReturn, VarianceCovariance
import time
import json
import config
import logging
from tools.Helpers import Parameters

SIM = "VarianceCovariance"
parser = argparse.ArgumentParser(description=f"Calculate the {SIM} risk forecasts (VaR/ES). For intense computations writes the batch-wise calculated risk forecasts in temp/portfolio. Afterwords, run `build_risk_dataframe.py` to aggregate the temporary files into a single DataFrame.")

parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the volatility forecasts.")

# Volatility model params
parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility porcesse's innovation distribution.")


parser.add_argument("-r", "--risk_metric",type=str,choices=config.RISKMETRICS,required=True,help="Choose the one-day ahead risk metric to be calculated.")
parser.add_argument("-a","--alpha",required=False,type=float,default=0.01,help="Set alpha for the desired alpha-level VaR/ES, default 0.01 for the 1%%-VaR/ES.")
parser.add_argument("--parallel",action="store_true",help="Wether to run the calculation concurrently or serial. Default: False. Info: Serial, due to less overhead, is faster for small portfolios.")

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
                "alpha":args.alpha,
                "risk_metric":[
                    args.risk_metric
                ]
            },
            "calculation": {
                "parallel":args.parallel
            }
        }

params = Parameters(_params)

print(f"Starting {args.simulation} {' & '.join(params.simulation.risk_metric)} with parameters:")
print(params)

logging.basicConfig(filename=f'./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{args.simulation} with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)

print("Importing data...")
returns = pd.read_parquet(f"data/{args.portfolio}/portfolio_returns.parquet")
volatilities = pd.read_parquet(f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/volatility_forecasts.parquet")
print("Done!")

print("Creating adjusted return windows...")
_, adj_r_windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
print("Done!")

# Creating the index for the one-day ahead VaR/ES forecasts
win_len = adj_r_windows.shape[1]
index = (returns.index.intersection(volatilities.index) + pd.offsets.BusinessDay(1))[win_len-1:]

del returns, volatilities

print(f"Calculating {args.simulation} {args.risk_metric}...")
start = time.perf_counter()

# -----------------
# START CALCULATION
# -----------------

match args.risk_metric:
    case "VaR":
        if args.parallel:
            print("Calculating concurrently...")
            with ProcessPoolExecutor() as p:
                results = list(tqdm(p.map(partial(VarianceCovariance.value_at_risk,alpha=args.alpha),adj_r_windows), total=len(adj_r_windows)))
        else:
            print("Calculating serial...")
            results = list(tqdm(map(partial(VarianceCovariance.value_at_risk,alpha=args.alpha),adj_r_windows), total=len(adj_r_windows)))
    
    case "ES":
        if args.parallel:
            print("Calculating concurrently...")
            with ProcessPoolExecutor() as p:
                results = list(tqdm(p.map(partial(VarianceCovariance.expected_shortfall,alpha=args.alpha),adj_r_windows), total=len(adj_r_windows)))
        else:
            print("Calculating serial...")
            results = list(tqdm(map(partial(VarianceCovariance.expected_shortfall,alpha=args.alpha),adj_r_windows), total=len(adj_r_windows)))
    
    case _ as e:
        raise NotImplementedError(f"Risk measure {e} not implemented!")


stop = time.perf_counter()
   
print("Done!")
print(f"Calulation finished in {stop - start:.2f} seconds.")

logging.info(f"Calulation finished in {stop-start:.2f} seconds.")

metrics = np.array(results)
forecasts = pd.Series(data=metrics,index=index,dtype=np.float32)

# Saving 
#level = int(100*ALPHA)
path = f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/{params.simulation.name}/"
with open(path+"params.json","wt") as f:
    json.dump(params.dict,f,indent=4)

forecasts.to_frame(name=args.risk_metric.lower()).to_parquet(path+f"{args.risk_metric}.parquet")
print(path+f"{args.risk_metric}.parquet written!")
print(path+"params.json written!")