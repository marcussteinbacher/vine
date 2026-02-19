import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from models import AdjustedReturn, HistoricalSimulation, VarianceCovariance
import time
import json
import config

parser = argparse.ArgumentParser(description="Calculate the risk forecasts. For intense computations writes the batch-wise calculated risk forecasts in temp/portfolio. Afterwords, run `build_risk_dataframe.py` to aggregate the temporary files into a single DataFrame.")

parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the volatility forecasts. Results will be stored in data/portfolio/...")
parser.add_argument("-m","--model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
parser.add_argument("-d","--distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the innovation's distribution model.")
parser.add_argument("-s","--simulation",type=str,choices=config.SIMULATIONS,required=True,help="Choose the risk resolution method.")
parser.add_argument("-r", "--risk_metric",type=str,choices=config.RISKMETRICS,required=True,help="Choose the one-day ahead risk metric to be calculated.")
parser.add_argument("-a","--alpha",required=False,type=float,default=0.01,help="Set alpha for the desired alpha-level VaR/ES, default 0.01 for the 1%%-VaR/ES.")

args = parser.parse_args()

PORTFOLIO = args.portfolio
MODEL = args.model
DIST = args.distribution
SIM = args.simulation
RISK = args.risk_metric
ALPHA = args.alpha

print("Starting calculation with parameters:")
print(json.dumps(vars(args), indent=2))

print("Importing data...")
returns = pd.read_parquet(f"data/{PORTFOLIO}/portfolio_returns.parquet")
volatilities = pd.read_parquet(f"data/{PORTFOLIO}/{MODEL}/{DIST}/volatility_forecasts.parquet")
print("Done!")


print("Creating adjusted return windows...")
_, adj_r_windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
print("Done!")

# Creating the index for the one-day ahead VaR/ES forecasts
win_len = adj_r_windows.shape[1]
index = (returns.index.intersection(volatilities.index) + pd.offsets.BusinessDay(1))[win_len-1:]

del returns, volatilities

print("Calculating risk metric!")
start = time.perf_counter()

match (SIM, RISK):
    case ("HistoricalSimulation","VaR"):
        # Computationally not very intensive, thus no need for batch-wise calculation 
        
        # Concurrently -> Overhead! Linear calculation efficient linear
        #with ProcessPoolExecutor() as p:
        #    results = list(tqdm(p.map(partial(HistoricalSimulation.value_at_risk,alpha=ALPHA),adj_r_windows), total=len(adj_r_windows)))

        # Linear
        results = list(tqdm(map(partial(HistoricalSimulation.value_at_risk,alpha=ALPHA),adj_r_windows), total=len(adj_r_windows)))

    case ("HistoricalSimulation","ES"):
        results = list(tqdm(map(partial(HistoricalSimulation.expected_shortfall,alpha=ALPHA),adj_r_windows), total=len(adj_r_windows)))
    
    case ("VarianceCovariance","VaR"):
        results = list(tqdm(map(partial(VarianceCovariance.value_at_risk,alpha=ALPHA),adj_r_windows), total=len(adj_r_windows)))
    
    case ("VarianceCovariance","ES"):
        results = list(tqdm(map(partial(VarianceCovariance.expected_shortfall,alpha=ALPHA),adj_r_windows), total=len(adj_r_windows)))

    case _ as e:
        raise NotImplementedError(f"Combination {e} not implemented!")

stop = time.perf_counter()
   
print("Done!")
print(f"Calulation finished in {stop - start} seconds.")

metrics = np.array(results)
forecasts = pd.Series(data=metrics,index=index,dtype=np.float32)

# Saving 
#level = int(100*ALPHA)

forecasts.to_pickle(f"data/{PORTFOLIO}/{MODEL}/{DIST}/{SIM}/{RISK}.pickle")