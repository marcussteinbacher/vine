import pyvinecopulib as pvc
from typing import Literal
from ._Simulation import value_at_risk, expected_shortfall


def get_family_counts(vc:pvc.Vinecop, how:Literal["tree","total"]="tree")->dict:
    """
    Returns a dictionary with the number of copula families per tree or total for a VineCopula.

    :param vc: pyvinecopulibVineCopula object
    :param how: str, either "tree" or "total". "tree" returns a dictionary with the number of copula families per tree. "total" returns a dictionary with the total number of copula families over all trees.

    :return: dict
    """
    counts = {}

    for i,tree in enumerate(vc.pair_copulas):
        counts[i] = {}
        for cop in tree:
            name = cop.family.name
            counts[i][name] = counts[i].get(name, 0) + 1
    
    if how == "tree":
        return counts
    
    else:
        keys = []
        for tree in counts:
            keys += list(counts[tree].keys())
        distinct = set(keys)

        total = {cop:0 for cop in distinct}

        for _,tree in counts.items():
            for fam,count in tree.items():
                total[fam] += count

    return counts



def build_family_set(families:list[str]):
    """
    Creates a list of pyvinecopulib.BicopFamily objects from a list of family str names.
    
    :param families: A subset of bivariate copula families. Possible: Gaussian, Student, Clayton, Frank, Gumbel, Joe, Independent.
    :type families: list[str]
    """
    d = {
        "Gaussian":pvc.BicopFamily.gaussian,
        "Student":pvc.BicopFamily.student,
        "Clayton":pvc.BicopFamily.clayton,
        "Frank":pvc.BicopFamily.frank,
        "Gumbel":pvc.BicopFamily.gumbel,
        "Joe":pvc.BicopFamily.joe,
        "Independent":pvc.BicopFamily.indep,
        "BB1":pvc.BicopFamily.bb1,
        "BB6":pvc.BicopFamily.bb6,
        "BB7":pvc.BicopFamily.bb7,
        "BB8":pvc.BicopFamily.bb8,
        "Tawn":pvc.BicopFamily.tawn,
        "TLL":pvc.BicopFamily.tll

    }
    return [d[family] for family in families] if families else []


def get_controls(fit_controls:pvc.FitControlsVinecop)->dict:
    """
    Extract the controls from a pyvinecopulib.FitControlsVinecop object and return them as a dictionary.
    """
    params = {}
    for k in filter(lambda s: not s.startswith("_") and s not in ("seeds","weights"), dir(fit_controls)):
        params[k] = getattr(fit_controls,k)
    params["family_set"] = [f.name for f in fit_controls.family_set]
    return params


class VineCopulaResult:
    def __init__(self, vine:pvc.Vinecop):
        self.vine = vine
    
    def __repr__(self):
        return f"VineCopula(dim={self.vine.dim},trunc_lvl={self.vine.trunc_lvl},threshold={self.vine.threshold})"