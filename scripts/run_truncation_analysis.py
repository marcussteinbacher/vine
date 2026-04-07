import numpy as np 
import pyvinecopulib as pvc
from models import AdjustedReturn
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import argparse
import config
from concurrent.futures import ProcessPoolExecutor

MODEL = "Garch"
DIST = "Empirical"
CONTROLS = pvc.FitControlsVinecop(
    parametric_method="itau",
    selection_criterion="mbicv", # Nagler et al., 2018
    select_trunc_lvl=True, # automatically select truncation level, computationally expensive, better set manually
    num_threads=1
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run regression anylysis of the truncation level for vine copulas. Saves a dict of dimension:[vine_copulas] to data/truncation/vinecopulas.pkl")
    parser.add_argument("-p","--portfolio", nargs='+',choices=config.PORTFOLIOS, default=[], type=int,required=True,help="Choose the portfolios to include in the regression analysis.")
    parser.add_argument("-n",type=int,required=False,default=20,help="Number of random windows to fit vinecopulas. Default: 20")
    return parser.parse_args()

def func(w):
    u = pvc.to_pseudo_obs(w)
    rvine = pvc.Vinecop.from_data(u, controls=CONTROLS)
    return rvine


def main():
    args = parse_args()

    d = {}

    for p in tqdm(args.portfolio,desc="Portfolio",leave=False):
        # Reading data
        returns = pd.read_parquet(f"data/{p}/portfolio_returns.parquet")
        volatilities = pd.read_parquet(f"data/{p}/{MODEL}/{DIST}/volatility_forecasts.parquet")

        window_indices, adj_r_windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)

        # Choosing N random windows
        r_w_i = np.random.choice(range(len(adj_r_windows)), args.n, replace=False)
        r = adj_r_windows[r_w_i]

        # Fitting vine copulas
        with ProcessPoolExecutor() as executor:
            vines = list(tqdm(executor.map(func,r), desc="Window", total=len(r),leave=False))

        d[p] = vines

    with open("data/truncation/vinecopulas.pkl", "wb") as f:
        pkl.dump(d, f)

    print("data/truncation/vinecopulas.pkl written!")

if __name__ == "__main__":
    main()