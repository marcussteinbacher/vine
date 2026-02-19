import json 
import argparse
import itertools


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
    
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest) or {}
        for item in values:
            k, v = item.split('=', 1)
            d[k] = self.parse_value(v)
        setattr(namespace, self.dest, d)