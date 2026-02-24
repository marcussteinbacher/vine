import pandas as pd
import glob
import argparse
import os
import json
from tools.Helpers import Parameters
import pickle as pkl


parser = argparse.ArgumentParser(description="Populate the batch-wise risk (VaR, ES, Model Information) forecasts in the temp folder. Info: As a safety hook the folder must be existent and be created manually before executing the command!")
parser.add_argument("--keep",action="store_true",help="Wether to keep the temporary files. Default: False, deletes temporary files after a successful operation.")
args = parser.parse_args()

# Read temp params json
params = Parameters.from_json("temp/params.json")

print(f"Populating temporary files for parameters:")
print(json.dumps(params.dict, indent=4))


def file_gen(files):
    """
    A generator to read all files from disk into a DataFrame.
    
    :param files: list[str] of files paths
    """
    for file in files:
        yield pd.read_parquet(file)


path = []
path.append(f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/{params.simulation.name}")
if params.simulation.name == "MultivariateCopula": 
    path.append(f"/{params.simulation.copula}")
path.append(f"/{params.simulation.margin_distribution}")
path = "".join(path)

# -------------
# VaR DataFrame
# -------------

def build_var_df(files):

    df = pd.concat(file_gen(files))
    df.sort_index(inplace=True)

    # Saving
    df.to_parquet(path+"/VaR.parquet")

    print(f"{path}/VaR.parquet written!")

    # Removing temporary files
    if not args.keep:
        for file in files:
            os.remove(file)

# -------------
# ES DataFrame
# -------------

def build_es_df(files):

    df = pd.concat(file_gen(files))
    df.sort_index(inplace=True)

    # Saving
    df.to_parquet(path+"/ES.parquet")

    print(f"{path}/ES.parquet written!")

    # Removing temporary files
    if not args.keep:
        for file in files:
            os.remove(file)

# -------------
# Aggregate risk_models.pkl
# -------------

def build_models_dict(files):

    d = {}
    for file in files:
        with open(file,"rb") as f:
            d.update(pkl.load(f))
    sorted_dict = dict(sorted(d.items()))

    with open(path+"/models.pkl","wb") as f:
        pkl.dump(sorted_dict,f)

    print(f"{path}/models.pkl written!")

    if not args.keep:
        for file in files:
            os.remove(file)


if __name__ == "__main__":
    files = glob.glob(f"temp/var_*.parquet")
    build_var_df(files)

    files = glob.glob(f"temp/es_*.parquet")
    build_es_df(files)

    files = glob.glob("temp/models_*.pkl")
    build_models_dict(files)

    # Move model params.pkl
    os.system(f"mv temp/params.json {path}/params.json")
    print(f"{path}/params.json written!")

# -------------
# Copula params DataFrame
# -------------

# Dont handle this anymore due to redundant information. 
# It containes information about the margin distributions and not the copula parameters
# due to its size.

# files = glob.glob(f"temp/{PORTFOLIO}/MVC_params_{MODEL}_{DIST}_{SIM}_{COPULA}_{MARGIN}_*.parquet")

# df = pd.concat(file_gen(files))
# df.sort_index(inplace=True)
# df.reset_index(inplace=True,drop=True)
# df.index.name = "window"

# # Saving
# df.to_parquet(f"data/{PORTFOLIO}/{MODEL}/{DIST}/{SIM}/{COPULA}/{MARGIN}/params.parquet")

# # Removing temporary files
# if not args.keep:
#     for file in files:
#         os.remove(file)

