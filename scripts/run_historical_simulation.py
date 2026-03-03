import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from models import AdjustedReturn
from simulations.HistoricalSimulation import simulate_hist
import time
import json
import config
import logging
from tools.Helpers import Parameters


_SIM = "HistoricalSimulation"
logging.basicConfig(filename='./log/risk_forecasts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description=f"Calculate the {_SIM} risk forecasts (VaR/ES). Directly writes the results into the simulation folder.")

    parser.add_argument("-p","--portfolio", choices=config.PORTFOLIOS, type=int,required=True,help="Choose the portfolio for which to calculate the volatility forecasts.")

    # Volatility model params
    parser.add_argument("-vm","--volatility_model", choices=config.VOLATILITYMODELS, type=str,required=True,help="Choose the volatility process for the constant mean model.")
    parser.add_argument("-id","--innovation_distribution", choices=config.INNOVATIONDISTRIBUTIONS, type=str,required=True,help="Choose the volatility processe's innovation distribution.")

    #parser.add_argument("-r", "--risk_metric",type=str,choices=config.RISKMETRICS,required=True,help="Choose the one-day ahead risk metric to be calculated.")
    parser.add_argument("-a","--alpha",required=False,type=float,default=0.01,help="Set alpha for the desired alpha-level VaR/ES, default 0.01 for the 1%%-VaR/ES.")
    #parser.add_argument("--parallel",action="store_true",help="Wether to run the calculation concurrently or serial. Default: False. Info: Serial, due to less overhead, is faster for small portfolios.")

    return parser.parse_args()

def main():
    args = parse_args()

    _params = {"portfolio":args.portfolio,
          "volatility": {
              "mean_model":"ConstantMean",
              "volatility_model":args.volatility_model,
              "innovation_distribution":args.innovation_distribution
            },
            "simulation": {
                "name":_SIM,
                "alpha":args.alpha,
                "risk_metric":[
                    "VaR",
                    "ES"
                ]
            },
            "calculation": {
                "parallel":False #args.parallel
            }
        }
    
    params = Parameters(_params)

    # Construct simulation path
    path_s = f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/{params.simulation.name}/"


    print(f"Starting {_SIM} {args.risk_metric} with parameters:")
    print(json.dumps(_params, indent=4))


    logging.info("STARTING CALCULATION\n"
             + "-" * 50
             + f"\n{_SIM} with parameters:"
             + "\n" + str(params)
             + "\n" + "-" * 50)


    print("Importing data...")
    returns = pd.read_parquet(f"data/{params.portfolio}/portfolio_returns.parquet")
    volatilities = pd.read_parquet(f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/volatility_forecasts.parquet")
    print("Done!")

    print("Creating adjusted return windows...")
    _, adj_r_windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
    print("Done!")

    # Creating the index for the one-day ahead VaR/ES forecasts
    index = returns.index + pd.offsets.BusinessDay(1)

    del returns, volatilities

    print(f"Calculating {_SIM} VaR & ES...")
    start = time.perf_counter()

    # -----------------
    # START CALCULATION
    # -----------------

    results = list(tqdm(map(partial(simulate_hist,alpha=params.simulation.alpha),adj_r_windows), total=len(adj_r_windows)))
    
    stop = time.perf_counter()
   
    print("Done!")
    print(f"Calulation finished in {stop-start:.2f} seconds.")

    logging.info(f"Calulation finished in {stop-start:.2f} seconds.")
    
    var, es = zip(*results)

    del results

    #vars = np.array(var, dtype=np.float32)
    #ess = np.array(es, dtype=np.float32)

    vars = pd.Series(data=var,index=index,dtype=np.float32)
    ess = pd.Series(data=es,index=index,dtype=np.float32)

    del var, es

    # Saving 
    vars.to_frame(name="var").to_parquet(path_s+"VaR.parquet")
    print(path_s+"VaR.parquet written!")

    ess.to_frame(name="es").to_parquet(path_s+"ES.parquet")
    print(path_s+"ES.parquet written!")

    with open(path_s+"/params.json","wt") as f:
        json.dump(params.dict,f,indent=4)
    
    print(path_s+"/params.json written!")


if __name__ == "__main__":
    main()