import json 
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import pickle as pkl
import subprocess


class Parameters:
    """
    A helper class to represent a nested dictionary to access all attributes in chained dot-notation.
    """
    
    def __init__(self,d:dict):
        self.dict = d
        for k,v in d.items():
            setattr(self,k,v)
            if isinstance(v,dict):
                setattr(self,k,Parameters(v))

    def __repr__(self):
        return json.dumps(self.dict, indent=2)
    
    @classmethod
    def from_json(cls,path:str):
        """
        Create a Parameters object from a json file.
        
        :param path: Path to the json file.
        :type path: str
        """
        return cls(json.load(open(path,"rt")))
    
    def __dir__(self)->tuple[str]:
        return tuple(self._all_keys(self.dict))

    def _all_keys(self, d:dict):
        """
        Generator that extracts all keys from a nested dictionary.
        """
        for k in d.keys():
            yield k
        for k in d.values():
            if isinstance(k, dict):
                yield from self._all_keys(k)


def chunks(iterable, chunk_size):
    """
    Generator that yields chunks of size chunk_size from an iterable.
    """
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def _get_path(params:Parameters)->str:
    """
    Extracts the simulation path from a tools.Parameters object.
    """
    base = f"data/{params.portfolio}/{params.volatility.volatility_model}/{params.volatility.innovation_distribution}/"
    
    if hasattr(params,"simulation"):
         # Its a simulation and not a volatility forcast
        match params.simulation.name:
            case "MultivariateCopula":
                path = base + f"{params.simulation.name}/{params.simulation.copula}/{params.simulation.margin_distribution}/"
            case "VineCopula":
                path = base + f"{params.simulation.name}/{params.simulation.margin_distribution}/"
            case _:
                path = base + f"{params.simulation.name}/"
    else:
        # Its a volatility forecast
        path = base

    return path


def save_scalars(data:dict, index:pd.Index|None=None,temp_dir:str="temp") -> None:
    """
    Takes {window_id: (scalar_a, scalar_b)} for all windows, sorted. E.g. from Runner(...).collect_scalars(),
    and creates a pandas Series with an optional index, e.g. portfolio_returns + pd.offsets.BusinessDay(1) for one-day ahead risk-forecasts.
    Saves VaR.parquet and ES.parquet into the simulation folder as specified in temp/params.json.
    """
    # Read params from temp
    params = Parameters.from_json(f"{temp_dir}/params.json")

    # Create path
    path = _get_path(params)
    
    # Concat to dataframe
    df = pd.DataFrame.from_dict(data,orient="index",columns=["var","es"],dtype=np.float32)
    if index is not None:
        df.index = index
    
    # Saving 
    df.loc[:,["var"]].to_parquet(path+"VaR.parquet")
    print(path+"VaR.prquet written!")

    df.loc[:,["es"]].to_parquet(path+"ES.parquet")
    print(path+"ES.parquet written!")


def save_objects(data:dict,temp_dir:str="temp") -> None:
    """
    Takes the collection {window_id: obj, ...} from Runner().collect_objects() and
    saves them into the simulation folder as of temp/params.json.
    """
    params = Parameters.from_json(f"{temp_dir}/params.json")
    path = _get_path(params)

    with open(path+"models.pkl","wb") as f:
        pkl.dump(data, f)

    print(path+"models.pkl written!")


def save_params(temp_dir:str="temp"):
    """
    Moves temp/params.json into the simulation folder.
    """
    params = Parameters.from_json(f"{temp_dir}/params.json")
    path = _get_path(params)
    subprocess.run(["mv", f"{temp_dir}/params.json", path+"params.json"])
    print(path+"params.json written!")


def save_vols(data:dict, index:pd.Index|None=None,temp_dir:str="temp") -> None:
    """
    Takes {window_id: (scalar_a, None)} for all windows, sorted. E.g. from Runner(...).collect_scalars(),
    and creates a pandas Series with an optional index, e.g. portfolio_returns + pd.offsets.BusinessDay(1) for one-day ahead risk-forecasts.
    Saves volatility_forecast.parquet into the simulation folder as specified in temp/params.json.
    """
    # Read params from temp
    params = Parameters.from_json(f"{temp_dir}/params.json")

    # Create path
    path = _get_path(params)
    
    # Concat to dataframe
    df = pd.DataFrame.from_dict(data,orient="index",dtype=np.float32)
    if index is not None:
        df.index = index
    
    # Saving 
    df.to_parquet(path+"volatility_forecasts.parquet")
    print(path+"volatility_forecasts.prquet written!")


def save_checkpoint(idx)->None:
    """
    Save the current progress when the calculation is interrupted. 
    """
    with open("temp/checkpoint","wt") as f:
        f.write(str(idx))


def get_checkpoint()->int:
    """
    Read the current progress when resuming an interrupted calculation. 
    Returns the index of the last successfully calculated batch.
    """
    if os.path.exists("temp/checkpoint"):
        with open("temp/checkpoint","rt") as f:
            check = f.read()
        return int(check)
    return 0


class StoreDict(argparse.Action):
    """
    Collect key=value pairs submitted in the terminal as a dictionary.
    """
    def parse_value(self,value):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            try:
                return int(value)
            except ValueError:
                return value  # fallback to string
    
    def __call__(self, parser, namespace, values:dict, option_string=None):
        d = getattr(namespace, self.dest)
        for item in values:
            k, v = item.split('=', 1)
            d[k] = self.parse_value(v)
        setattr(namespace, self.dest, d)