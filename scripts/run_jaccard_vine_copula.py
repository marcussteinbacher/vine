import pyvinecopulib as pvc
import pickle as pkl
from tqdm import tqdm
import os
import json
import pandas as pd
from models import AdjustedReturn, VineCopula
from simulations.VineCopula import simulate_vc_jaccard
import time
from functools import partial
from models.VineCopula import get_structure_changes
from tools.Runner import Runner
from tools.Helpers import Parameters
import multiprocessing as mp

# Force spawn
mp.set_start_method("spawn", force=True)


# ---------------------------------------------
_SIM = "Jaccard Vine Copula"
portfolio = 50
volatility_model = "Garch"
innovation_distribution = "Empirical"
margin_distribution = "Empirical"
stride = 250
trunc_lvl = None #25 # None No truncation, fit full vine
# ---------------------------------------------


path = f"data/{portfolio}/{volatility_model}/{innovation_distribution}/VineCopula/{margin_distribution}/Jaccard/"

fit_controls = pvc.FitControlsVinecop(
    parametric_method="itau",
    family_set=[
        pvc.indep,
        pvc.student, # symmetric tail dependence
        pvc.clayton, # lower tail dependence
        pvc.gumbel, # upper tail dependence
        pvc.joe, # symmetric tail dependence, stronger than student
    ],
    allow_rotations=True,
    selection_criterion="mbicv", # modified BIC for vine copulas
    preselect_families=True, # whether to exclude families based on symmetry of the data
    trunc_lvl=trunc_lvl if trunc_lvl is not None else portfolio,
    select_trunc_lvl=False,
    select_threshold=True, # automatically select threshold for thresholded vines
    num_threads=1
)

# Collecting all arguments into params
_params = {"portfolio":portfolio,
      "volatility": {
          "mean_model":"ConstantMean",
          "volatility_model":volatility_model,
          "innovation_distribution":innovation_distribution
        },
        "simulation": {
            "name":_SIM,
            "margin_distribution":margin_distribution,
            "n":100_000,
            "fit_method":"itau",
            "alpha":0.01,
            "risk_metric":[
                "VaR",
                "ES"
            ]
        },
        "calculation": {
            "save_freq":stride,
            "max_workers":os.cpu_count(),
        },
        "controls": VineCopula.get_controls(fit_controls)
    }

params = Parameters(_params)

# Save params.json to temp
with open("temp/params.json","wt") as f:
    json.dump(_params,f,indent=4)

def main():
    print("Importing data...")
    returns = pd.read_parquet(f"data/{portfolio}/portfolio_returns.parquet")
    volatilities = pd.read_parquet(f"data/{portfolio}/{volatility_model}/{innovation_distribution}/volatility_forecasts.parquet")
    print("Done!")

    print("Creating adjusted return windows...")
    windows_indices, windows = AdjustedReturn.adjusted_return_windows(returns, volatilities)
    print("Done!")

    fut_index = (returns.index + pd.offsets.BusinessDay(1))

    # Load Jaccard distances from reference portfolio
    with open("data/10/Garch/Empirical/VineCopula/Empirical/Jaccard/jaccard_distances.pkl","rb") as f:
        dists = pkl.load(f)

    objects = {}
    scalars = []

    start_time = time.perf_counter()

    # PASS 1: Identify where structures need to be computed
    print("Identifying structure change points...")

    structure_indices = get_structure_changes(dists, threshold=0.0)

    print(f"Found {len(structure_indices)} structure change points")

    # PASS 2: Pre-compute structures sequentially
    print("Pre-computing vine copula structures...")
    structures_dict = {}  # Maps index -> vine structure

    # Only compute structure, not families
    struct_controls = pvc.FitControlsVinecop(
        trunc_lvl=trunc_lvl if trunc_lvl is not None else portfolio,
        family_set=[pvc.BicopFamily.indep],
        num_threads=8
        )  

    for idx in tqdm(structure_indices, desc="Computing Structures"):
        pobs = pvc.to_pseudo_obs(windows[idx])
        vc_res = pvc.Vinecop.from_data(pobs, controls=struct_controls)
        structures_dict[idx] = vc_res.structure

    print("Done pre-computing structures!")

    # Calculate in parallel
    # Build the callable
    calc_fn = partial(simulate_vc_jaccard,
                    structure_dict=structures_dict, 
                    controls=fit_controls,
                    margin_dist=margin_distribution
                    )

    # Instantiate the runner
    runner = Runner(
        calculation_fn=calc_fn,
        data=windows,
        pass_indexed_window=True,
        n_workers=8,
        max_in_flight=8 * 8,
        flush_threshold=16, # flush scalars every 16 completed windows
        object_stride=stride, # save the model every 250 windows
        object_flush_threshold=4, # flush objects when 4 are enqueued
        temp_dir="temp"
    )

    print(f"Calculating {_SIM} VaR & ES...")
    start_time = time.perf_counter()

    runner.run()

    end_time = time.perf_counter()
    print(f"Calculation completed in {end_time-start_time:.2f} seconds!")

    # Collecting results
    scalars = runner.collect_scalars()
    objects = runner.collect_objects()

    # Save aggregated data: VaR.parquet, ES.parquet, models.pkl, params.json

    index = fut_index[-len(scalars):]
    var_series = pd.Series([v[0][0] for v in scalars.values()],index=index) # (N,) 
    es_series = pd.Series([v[1][0] for v in scalars.values()],index=index) # (N,)

    var_series.to_frame(name="var").to_parquet(path+"VaR.parquet")
    print(path+"VaR.parquet written!")

    es_series.to_frame(name="es").to_parquet(path+"ES.parquet")
    print(path+"ES.parquet written!")

    with open(path+"models.pkl","wb") as f:
        pkl.dump(objects, f)
    print(path+"models.pkl written!")

    params.calculation.runtime = end_time - start_time

    with open(path+"params.json","wt") as f:
        json.dump(params.dict,f,indent=2)
    print(path+"params.json written!")

if __name__ == "__main__":
    print(f"Starting {_SIM} VaR & ES with parameters:")
    print(params)
    main()


# PASS 3: Assign each window to its structure
# window_structure_map = {}
# last_structure_idx = None
# 
# for i in range(len(dists)):
# if i in structures_dict:
    # last_structure_idx = i
# window_structure_map[i] = last_structure_idx
# 
# PASS 4: Fit and sample in parallel
# def process_window_parallel(i, jac, win, structure_idx, stride):
    # """Process a single window with its pre-computed structure"""
    # if structure_idx is None:
        # First window, fit without structure
        # vc_res, var, es = simulate_vc(win, fit_controls, "Empirical")
    # else:
        # Use pre-computed structure
        # vc_res, var, es = simulate_vc(win, fit_controls, "Empirical", structure=structures_dict[structure_idx])
    # 
    # store_model = (i // stride == 0)
    # return i, var, es, vc_res.vine if store_model else None
# 
# print("Fitting and sampling in parallel...")
# results = Parallel(n_jobs=8)(
    # delayed(process_window_parallel)(i, jac, windows[i], window_structure_map[i], stride)
    # for i, jac in tqdm(enumerate(dists), desc="Window", total=len(dists))
# )
# 
# end_time = time.perf_counter()
# print(f"Calculation completed in {end_time - start_time:.2f} seconds!")
# 
# #Collect results in original order
# for i, var, es, model in sorted(results):
    # scalars.append((var, es))
    # if model is not None:
        # objects[i] = model
# 
# var, es = zip(*scalars)
# 
# index = fut_index[-len(var):]
# vars = pd.Series(data=var,index=index,dtype=np.float32)
# ess = pd.Series(data=es,index=index,dtype=np.float32)
# 
# vars.to_frame(name="var").to_parquet(f"data/{portfolio}/Garch/Empirical/VineCopula/Empirical/Jaccard/VaR.parquet")
# print("VaR written!")
# 
# ess.to_frame(name="es").to_parquet(f"data/{portfolio}/Garch/Empirical/VineCopula/Empirical/Jaccard/ES.parquet")
# print("ES written!")
# 
# with open(f"data/{portfolio}/Garch/Empirical/VineCopula/Empirical/Jaccard/models.pkl","wb") as f:
    # pkl.dump(objects, f)
# print("Models written!")