import json 
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import pickle as pkl
import subprocess
import pyvinecopulib as pvc


class Parameters:
    """
    A helper class to represent a nested dictionary to access all attributes in chained dot-notation.
    """
    
    def __init__(self,d:dict):
        self.dict = d
        for k,v in d.items():
            setattr(self,k,v)

    def __setattr__(self, key, value):
        if key == "dict":
            object.__setattr__(self, key, value)
            return

        if isinstance(value, Parameters):
            attr_value = value
            dict_value = value.dict
        elif isinstance(value, dict):
            attr_value = Parameters(value)
            dict_value = attr_value.dict
        else:
            attr_value = value
            dict_value = value

        object.__setattr__(self, key, attr_value)

        d = self.__dict__.get("dict")
        if isinstance(d, dict):
            d[key] = dict_value

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


def save_objects(data:dict,temp_dir:str="temp",folder:str="") -> None:
    """
    Takes the collection {window_id: obj, ...} from Runner().collect_objects() and
    saves them into the simulation folder as of temp/params.json.
    """
    params = Parameters.from_json(f"{temp_dir}/params.json")
    path = _get_path(params)
    path = path if not folder else path+folder+"/"

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
                try:
                    return float(value)
                except ValueError:
                    return value  # fallback to string
    
    def __call__(self, parser, namespace, values:dict, option_string=None):
        d = getattr(namespace, self.dest)
        for item in values:
            k, v = item.split('=', 1)
            d[k] = self.parse_value(v)
        setattr(namespace, self.dest, d)
 
 
def _decode_edge(matrix: np.ndarray, d: int, t: int, e: int) -> tuple:
    """
    Decode the conditioned pair and conditioning set for edge e in tree t.
 
    Parameters
    ----------
    matrix : np.ndarray, shape (d, d)
        Structure matrix from vine.structure.matrix (1-indexed labels).
    d : int
        Dimension of the vine.
    t : int
        Tree level, 0-indexed (Tree 1 = t=0, Tree 2 = t=1, ...).
    e : int
        Edge index within the tree, 0-indexed.
 
    Returns
    -------
    conditioned : frozenset of two ints (1-indexed variable labels)
    conditioning : frozenset of ints (1-indexed variable labels)
    """
    var1 = int(matrix[d - 1 - e, e])   # counter-diagonal element of column e
    var2 = int(matrix[t, e])            # element at row t of column e
    conditioning = frozenset(int(matrix[k, e]) for k in range(0, t))
    return frozenset({var1, var2}), conditioning
 
 
def get_pair_copula(
    vine: pvc.Vinecop,
    conditioned: tuple[int, int],
    conditioning: tuple[int, ...] = (),
    re_index: bool = True
) -> pvc.Bicop:
    """
    Extract a pair copula from a fitted Vinecop by its conditioned and
    conditioning variable sets (1-indexed, matching pyvinecopulib convention).
 
    Parameters
    ----------
    vine : pv.Vinecop
        A fitted vine copula object.
    conditioned : tuple of two ints
        The conditioned variable pair, e.g. (2, 5). Order does not matter.
        Variables are 1-indexed.
    conditioning : tuple of ints, optional
        The conditioning set, e.g. (1, 3). Order does not matter.
        Empty tuple for Tree 1 edges (default).
    re_index : bool, optional
        If True, re-index the pair copula to account for pyvinecopulibs indexing starting at 1.
 
    Returns
    -------
    tuple of (int, int, pvc.Bicop)
        The tree index, edge index, and the pair copula object.
 
    Raises
    ------
    ValueError
        If no matching edge is found. Call print_vine_edges(vine) to inspect
        all available edges.
 
    Examples
    --------
    # Tree 1 edge between variables 2 and 5:
    bc = get_pair_copula(vine, conditioned=(2, 5))
 
    # Tree 2 edge between variables 1 and 4, conditioned on variable 3:
    bc = get_pair_copula(vine, conditioned=(1, 4), conditioning=(3,))
 
    # Tree 3 edge between variables 2 and 6, conditioned on {1, 3}:
    bc = get_pair_copula(vine, conditioned=(2, 6), conditioning=(1, 3))
    """
    if re_index:
        # Re-index to match pyvinecopulib's 1-indexing convention
        conditioned = tuple(i + 1 for i in conditioned)
        conditioning = tuple(i + 1 for i in conditioning)

    target_conditioned  = frozenset(conditioned)
    target_conditioning = frozenset(conditioning)
 
    struct = vine.structure
    d      = struct.dim          # correct attribute: .dim, not .d
    matrix = struct.matrix       # d x d array, 1-indexed variable labels
    pair_copulas = vine.pair_copulas  # nested list [tree][edge]
 
    # tree t is 0-indexed (t=0 → Tree 1), edge e is 0-indexed within the tree.
    # At tree t there are d-1-t edges: e = 0, 1, ..., d-2-t.
    for t in range(d - 1):
        for e in range(d - 1 - t):
            edge_conditioned, edge_conditioning = _decode_edge(matrix, d, t, e)
            if (edge_conditioned  == target_conditioned and
                    edge_conditioning == target_conditioning):
                return t,e, pair_copulas[t][e]
 
    raise ValueError(
        f"No edge found with conditioned={set(conditioned)} and "
        f"conditioning={set(conditioning)}.\n"
        f"Call print_vine_edges(vine) to see all available edges."
    )


def get_common_edges(vine0:pvc.Vinecop, vine1:pvc.Vinecop, tree:int):
    """
    Extract the common edges between two vine copulas at a specific tree level.
    """
    from stats.Jaccard import VineStructureSnapshot

    edges0 = VineStructureSnapshot.from_vinecop(vine0).get_edges()[tree]
    edges1 = VineStructureSnapshot.from_vinecop(vine1).get_edges()[tree]
    return edges0 & edges1