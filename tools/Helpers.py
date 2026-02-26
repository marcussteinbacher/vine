import json 
import argparse
import itertools
import os


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