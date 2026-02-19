import pandas as pd
import glob
import argparse
import os
import config
import json
import numpy as np 
import pickle as pkl


parser = argparse.ArgumentParser(description="Populate the batch-wise volatility forecasts in the temp folder into a single DataFrame. Writes the actual volatility forecasts and model summary into data/{portfolio}/{model}/{distribution}. Info: As a double-check safety hook the folder must be existent and be created manually before executing the command!")
parser.add_argument("--keep",action="store_true",help="Wether to keep the temporary files. Default: False, deletes temporary files after a successful operation.")

args = parser.parse_args()

# Read temp params json
with open("temp/params.json","rt") as f:
    temp_args = json.load(f)

# TODO
args.portfolio = temp_args["portfolio"]
args.volatility_model = temp_args["volatility"]["volatility_model"]
args.volatility_distribution = temp_args["volatility"]["innovation_distribution"]

path=f"data/{args.portfolio}/{args.volatility_model}/{args.volatility_distribution}"

print(f"Building DataFrame for parameters:")
print(json.dumps(vars(args), indent=4))

# --------------------
# VOLATILITY FORECASTS
# --------------------

# Reading all temporary files
files = glob.glob(f"temp/volatility_forecasts_*.parquet")

def file_gen(files):
    """
    A generator to read all files from disk into a DataFrame.
    
    :param files: list[str] of files paths
    """
    for file in files:
        yield pd.read_parquet(file)

df = pd.concat(file_gen(files))
df.sort_index(inplace=True)


# Saving
df.to_parquet(f"{path}/volatility_forecasts.parquet")

print(f"{path}/volatility_forecasts.parquet written!")

# Removing temporary files
if not args.keep:
    for file in files:
        os.remove(file)

del df

# -----------------
# VOLATILITY MODELS
# -----------------

files = sorted(glob.glob(f"temp/volatility_models_summary_*.pkl"))

#df_models = pd.concat(file_gen(files),ignore_index=True)

dicts = [pkl.load(open(file,"rb")) for file in files]

keys = set()
for d in dicts:
    for k in d.keys():
        keys.add(k)

models_summary = {}

for k in keys:
    models_summary[k] = 0
    for d in dicts:
        if k not in d.keys():
            continue
        models_summary[k] += d[k]

json.dump(models_summary,open(f"{path}/volatility_models_summary.json","wt"),indent=4)
print(f"{path}/volatility_models_summary.json written!")

#def get_vol_model(s:str):
#    """
#    Extracts the core-volatility model, e.g. EGARCH from str s = <Constant Mean:EGARCH:Normal at 123298554051504>
#    
#    :param s: str
#    """
#    return s.split(":")[1] if s is not None else None
#
#df_models = df_models.map(get_vol_model)

#df_models.index.name = "window"

# Calculating statistics
#flat = df_models.map(lambda s: json.loads(s)["volatility"]).to_numpy().flatten()
#counts = {model:None for model in set(flat)}

#for key in counts.keys():
#    counts[key] = int(np.count_nonzero(df_models.map(lambda s: json.loads(s)["volatility"]).to_numpy()==key))


# Saving 
#with open(f"data/{args.portfolio}/{args.volatility_model}/{args.volatility_distribution}/volatility_models_summary.pkl","wb") as f:
#    pkl.dump(counts, f)

#print(f"data/{args.portfolio}/{args.volatility_model}/{args.volatility_distribution}/volatility_models_summary.pkl written!")

#df_models.to_parquet(f"data/{PORTFOLIO}/{MODEL}/{DIST}/volatility_models.parquet")

# Removing temporary files
if not args.keep:
    for file in files:
        os.remove(file)

# Move params.pkl
os.system(f"mv temp/params.json {path}/params.json")
print(f"{path}/params.json written!")