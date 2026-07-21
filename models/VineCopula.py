import pyvinecopulib as pvc
from typing import Literal
import numpy as np
import warnings
import networkx as nx
from scipy.stats import kendalltau
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from itertools import combinations
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

        return total


def get_empirical_trunc_lvl(vine:pvc.Vinecop)->int:
    """
    Returns the empirical truncation level of the vine, i.e. the tree level from which onwards
    there's only independent copulas.
    """
    emp_trunc = vine.trunc_lvl
    family_counts = get_family_counts(vine)
    for tree in list(family_counts.keys())[::-1]:
        if "indep" in family_counts[tree]:
            if len(family_counts[tree]) == 1:
                emp_trunc = tree
            else:
                break
    return emp_trunc


def get_frequency_trunc_lvl(vine:pvc.Vinecop, thresh:float=0.9)->int:
    """
    Returns the empirical truncation level of the vine, i.e. the tree level for which the percentage of
    independent copula exceeds thresh for the first time.
    """
    
    emp_trunc = vine.trunc_lvl
    family_counts = get_family_counts(vine)

    # Get family percentage per tree
    for lvl,d in family_counts.items():
        for f,c in d.items():
            family_counts[lvl][f] = c/(emp_trunc-lvl)
            if f == "indep" and c/(emp_trunc-lvl) > thresh:
                return lvl
    return emp_trunc


def build_family_set(families:list[str]):
    """
    Creates a list of pyvinecopulib.BicopFamily objects from a list of family str names.
    
    :param families: A subset of bivariate copula families. Possible: Gaussian, Student, Clayton, Frank, Gumbel, Joe, Independent.
    :type families: list[str]
    """
    d = {
        "Gaussian":pvc.gaussian, # pvc.BicopFamily.gaussian
        "Student":pvc.student,
        "Clayton":pvc.clayton,
        "Frank":pvc.frank,
        "Gumbel":pvc.gumbel,
        "Joe":pvc.joe,
        "Independent":pvc.indep,
        "BB1":pvc.bb1,
        "BB6":pvc.bb6,
        "BB7":pvc.bb7,
        "BB8":pvc.bb8,
        "Tawn":pvc.tawn,
        "TLL":pvc.tll,  
    }
    return [d[family] for family in families] if families else []


def get_controls(fit_controls:pvc.FitControlsVinecop|pvc.FitControlsBicop)->dict:
    """
    Extract the controls from a pyvinecopulib.FitControlsVinecop object and return them as a dictionary.
    """
    params = {}
    for k in filter(lambda s: not s.startswith("_") and s not in ("seeds","weights"), dir(fit_controls)):
        params[k] = getattr(fit_controls,k)
    params["family_set"] = [f.name for f in fit_controls.family_set]
    return params


def extract_controlsbicop(controlsvinecop:pvc.FitControlsVinecop):
    """
    Extract the controls from a pyvinecopulib.FitControlsVinecop object and return only 
    the FitControlsBicop parameters.
    """
    bi_keys = get_controls(pvc.FitControlsBicop()).keys()
    
    return pvc.FitControlsBicop(**{k:controlsvinecop.__getattribute__(k) for k in bi_keys})


def get_structure_changes(dists:list[float],threshold=0.0):
    """
    Identify significant regime changes where we need to re-calculate the structure.
    Returns a list of indices where the distance exceeds the threshold and 
    the structur needs to be calculated.

    **Arguments**
    - dists: weighted jaccard distance between consecutive vine copulas
    - threshold: the minimum distance required to identify a regime change

    **Returns**
    - change_points: list of indices where the structure needs to be re-calculated
    """
    change_points = [0] # Always evaluate the first structure
    for i in range(1, len(dists)):
        if dists[i] > threshold:
            change_points.append(i)
    return change_points


class VineCopulaResult:
    def __init__(self, vine:pvc.Vinecop, threshold=None):
        self.vine = vine
        self.threshold=threshold # threshold cannot be pickles by vinecopulib, so we store it separately
    
    def __repr__(self):
        return f"VineCopula(dim={self.vine.dim},trunc_lvl={self.vine.trunc_lvl},threshold={self.threshold})"
    
# -----------------------
# Tail-thresholded Kendall's tau and related functions for building a custom vine structure.
# -----------------------

def threshold_kendalls_tau(u, v, threshold=0.25, tail='lower'):
    """
    Computes Kendall's tau restricted strictly to specified tail quantiles
    of the pseudo-observations.
    """
    if tail == 'lower':
        mask = (u < threshold) & (v < threshold)
    elif tail == 'upper':
        mask = (u > 1 - threshold) & (v > 1 - threshold)
    elif tail == "both":
        mask_lower = (u < threshold) & (v < threshold)
        mask_upper = (u > 1 - threshold) & (v > 1 - threshold)
        mask = mask_lower | mask_upper
    else:
        raise ValueError("Tail must be 'lower','upper', or 'both'.")
        
    if np.sum(mask) < 10:  # Stability safeguard for extreme thresholds, less than 10 pairs of extreme events
        return 0.0
        #tau, _ = kendalltau(u, v)  # Fall back to standard global tau if tail is too sparse 
    else:
        tau, _ = kendalltau(u[mask], v[mask])
    return np.nan_to_num(tau)


class VineTreeEdge:
    """
    Explicit tracker for an R-Vine edge across arbitrary tree depths.
    Uses Python object hashing to inherently enforce the Proximity Condition.
    """
    def __init__(self, conditioned, conditioning, copula, h_dict, parents=None):
        self.conditioned = set(conditioned)   # e.g., {1, 3}
        self.conditioning = set(conditioning) # e.g., {2, 4}
        self.copula = copula                 # Fitted pv.Bicop object
        self.h_dict = h_dict                 # Maps var_idx -> conditional data array
        self.parents = set(parents) if parents is not None else set()
        self.all_vars = self.conditioned | self.conditioning


def compute_tail_tau_matrix(data, threshold=0.25, tail='lower'):
    d = data.shape[1]
    matrix = np.ones((d, d))
    for i, j in combinations(range(d), 2):
        tau = threshold_kendalls_tau(data[:, i], data[:, j], threshold=threshold, tail=tail)
        matrix[i, j] = matrix[j, i] = np.abs(tau)
    return matrix


def fit_custom_tail_vine(u_data, controls:pvc.FitControlsBicop, trunc_lvl, threshold, tail_quantile=0.25, tail='lower'):
    """
    Fit a custom vine structure using the tail-thresholded Kendall's tau.
    """
    n, d = u_data.shape

    tail_tau_matrix = compute_tail_tau_matrix(u_data, threshold=tail_quantile, tail=tail)
    
    # Store all fitted edges grouped by tree level
    vines_by_tree = {}

    # Store all tail-weighted taus by tree level for post-hoc analysis
    taus_by_tree = {}
    
    # --- TREE 1 -----------------------------------------------------------
    
    G1 = nx.Graph()
    for i in range(d):
        for j in range(i + 1, d):
            #tau_u = threshold_kendalls_tau(u_data[:, i], u_data[:, j], threshold, tail)

            tau_u = tail_tau_matrix[i, j]  # Use precomputed matrix for efficiency
            
            # DEBUG: Print the exact weights the MST is about to evaluate
            #if (i == 4 and j == 5) or (i == 4 and j == 8) or (i == 5 and j == 8):
            #    print(f"Internal weight for ({i}, {j}): {np.abs(tau_u)}")

            G1.add_edge(i, j, weight=np.abs(tau_u),tailtau=tau_u)
            
    mst1 = nx.maximum_spanning_tree(G1, weight='weight',algorithm="prim")
    
    current_layer_edges = []
    current_layer_tailtaus = []

    for u, v, attr in mst1.edges(data=True):
        pair_data = u_data[:, [u, v]]

        tailtau = attr["tailtau"]
        current_layer_tailtaus.append(tailtau)

#--> HERE
        # If tailtau <= threshold, use indep, else use bicop
        if np.abs(tailtau) <= threshold:
            bicop = pvc.Bicop(pvc.indep)
            #print("TAILTAU BELOW THRESHOLD, USING INDEP")
        else:
            bicop = pvc.Bicop()
            bicop.select(pair_data, controls)
        
        # hfunc1: u given v | hfunc2: v given u
        h1 = bicop.hfunc1(pair_data)  
        h2 = bicop.hfunc2(pair_data)  
        
        edge_obj = VineTreeEdge(
            conditioned={u, v},
            conditioning=set(),
            copula=bicop,
            h_dict={u: h1, v: h2},
            parents={u, v} # Use initial variable indices as unique parent trackers
        )
        current_layer_edges.append(edge_obj)
        
    vines_by_tree[1] = current_layer_edges
    taus_by_tree[1] = current_layer_tailtaus

    # --- TREES 2 to d-1 ---------------------------------------------------
    for tree_idx in range(2, d):
        
        G_curr = nx.Graph()
        num_nodes = len(current_layer_edges)
        
        # Step A: Build Candidate Graph following the Proximity Condition
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_A = current_layer_edges[i]
                edge_B = current_layer_edges[j]
                
                # Proximity Condition: Edges must share exactly one node in the previous tree
                if len(edge_A.parents & edge_B.parents) == 1:
                    # Isolate conditioned vs conditioning variables
                    common_vars = edge_A.all_vars & edge_B.all_vars
                    var_A = list(edge_A.all_vars - common_vars)[0]
                    var_B = list(edge_B.all_vars - common_vars)[0]
                    
                    # Pull respective h-functions mapping to the isolated variables
                    u_A = edge_A.h_dict[var_A]
                    u_B = edge_B.h_dict[var_B]
                    
                    # Compute custom localized tail-rank correlation
                    tau_u = threshold_kendalls_tau(u_A, u_B, threshold, tail)
                    
                    # Cache structural properties in the edge attributes
                    G_curr.add_edge(i, j, weight=np.abs(tau_u), 
                                    var_A=var_A, var_B=var_B, common_vars=common_vars, tailtau=tau_u)
                    
        if len(G_curr.edges()) == 0:
            print(f"No valid proximity-conditioned edges available. Truncating at Tree {tree_idx-1}.")
            break
            
        # Step B: Maximum Spanning Tree Optimization
        mst_curr = nx.maximum_spanning_tree(G_curr, weight='weight', algorithm="prim")
        
        next_layer_edges = []
        next_layer_tailtaus = []

        for idx_A, idx_B, attr in mst_curr.edges(data=True):
            edge_A = current_layer_edges[idx_A]
            edge_B = current_layer_edges[idx_B]
            
            var_A, var_B = attr['var_A'], attr['var_B']
            common_vars = attr['common_vars']
            next_layer_tailtaus.append(attr['tailtau'])

            #print(edge_A.h_dict.keys(), var_A, var_B)

            # Step B: Dynamically resolve which variable belongs to which parent edge
            if var_A in edge_A.h_dict and var_B in edge_B.h_dict:
                u_A = edge_A.h_dict[var_A]
                u_B = edge_B.h_dict[var_B]
            elif var_B in edge_A.h_dict and var_A in edge_B.h_dict:
                # The edges are swapped relative to var_A/var_B; route them correctly
                u_A = edge_A.h_dict[var_B]
                u_B = edge_B.h_dict[var_A]
            else:
                raise KeyError(
                    f"Structural Mismatch: Expected to find conditioned variables ({var_A}, {var_B}) "
                    f"in parent edges. edge_A has keys {list(edge_A.h_dict.keys())}, "
                    f"edge_B has keys {list(edge_B.h_dict.keys())}."
                )
            
            # Step C: Pair Copula Selection via pyvinecopulib
            pair_data = np.column_stack([u_A, u_B])

# --> HERE
            # If tailtau <= threshold, use independence copula; else fit a bicopula
            if np.abs(attr['tailtau']) <= threshold:
                bicop = pvc.Bicop(pvc.indep)
                #print("TAILTAU BELOW THRESHOLD, USING INDEP")
            else:
                bicop = pvc.Bicop()
                bicop.select(pair_data, controls)
            
            # If trunc_lvl reached use independence copula
            if tree_idx > trunc_lvl:
                bicop = pvc.Bicop(pvc.indep) 
            else:
                bicop = pvc.Bicop()
                bicop.select(pair_data, controls)
            
            # Compute conditional transformations for the next tree level
            h1 = bicop.hfunc1(pair_data)  # var_A given var_B and common_vars
            h2 = bicop.hfunc2(pair_data)  # var_B given var_A and common_vars
            
            new_edge = VineTreeEdge(
                conditioned={var_A, var_B},
                conditioning=common_vars,
                copula=bicop,
                h_dict={var_A: h1, var_B: h2},
                parents={edge_A, edge_B} # The objects themselves act as unique parent keys
            )
            next_layer_edges.append(new_edge)
            
        vines_by_tree[tree_idx] = next_layer_edges
        taus_by_tree[tree_idx] = next_layer_tailtaus
        current_layer_edges = next_layer_edges

    return vines_by_tree, taus_by_tree


class CustomVineEdge:
    def __init__(self, conditioned, conditioning, copula_object):
        self.conditioned_variables = set(conditioned)   # e.g., {1, 3}
        self.conditioning_variables = set(conditioning) # e.g., {2, 4}
        self.copula = copula_object


def get_custom_trees(custom_vine):
    """
    Converts the custom vine structure into a list of lists of CustomVineEdge objects.
    
    Parameters:
    -----------
    custom_vine : dict
        Dictionary where keys are tree levels (1-based) and values are lists of VineTreeEdge objects.
        
    Returns:
    --------
    custom_trees : list of lists of CustomVineEdge
        Each inner list corresponds to a tree level, containing CustomVineEdge objects.
    """
    custom_trees = []
    for level, edges in custom_vine.items():
        tree_edges = []
        for edge in edges:
            tree_edges.append(CustomVineEdge(edge.conditioned, edge.conditioning, edge.copula))
        custom_trees.append(tree_edges)
    return custom_trees


def translate_custom_vine(custom_trees):
    """
    Translates a custom-built vine tree sequence into components
    compatible with pyvinecopulib.Vinecop.from_structure.
    
    Parameters:
    -----------
    custom_trees : list of lists of CustomVineEdge
        Length d-1 list. custom_trees[0] contains Tree 1 edges (no conditioning),
        custom_trees[1] contains Tree 2 edges (1 conditioning variable), etc.
    """

    d = len(custom_trees) + 1  # Number of variables is one more than edges in Tree 1

    # Initialize an empty d x d matrix
    matrix = np.zeros((d, d), dtype=int)
    
    # Initialize the nested list structure for pair-copulas
    pair_copulas = [[None] * (d - 1 - t) for t in range(d - 1)]
    
    used_edges = set()
    placed_diagonals = set()
    
    # Construct column by column from left to right (e = 0 to d-2)
    for e in range(d - 1):
        t_top = d - 2 - e  # The highest tree index providing an edge for this column
        
        # Find the single unused edge in the highest available tree
        unused_top_edges = [edge for edge in custom_trees[t_top] if edge not in used_edges]
        if not unused_top_edges:
            raise ValueError(f"No available unique edge found in Tree {t_top + 1} for column {e}.")
        
        edge_top = unused_top_edges[0]
        cond_vars = list(edge_top.conditioned_variables)
        
        chosen_bottom = None
        path_edges = []
        path_vars = []
        
        # Determine which conditioned variable serves as the structural pivot (bottom element)
        for candidate_bottom in cond_vars:
            current_bottom = candidate_bottom
            current_top_var = [v for v in cond_vars if v != current_bottom][0]
            
            temp_path_edges = [edge_top]
            temp_path_vars = [current_top_var]
            current_conditioning = edge_top.conditioning_variables
            
            success = True
            # Step down through the lower trees to build the path
            for t in range(t_top - 1, -1, -1):
                found_edge = None
                for edge in custom_trees[t]:
                    if (edge not in used_edges and 
                        current_bottom in edge.conditioned_variables and 
                        edge.conditioning_variables.issubset(current_conditioning)):
                        found_edge = edge
                        break
                
                if found_edge is None:
                    success = False
                    break
                
                other_var = [v for v in found_edge.conditioned_variables if v != current_bottom][0]
                temp_path_edges.append(found_edge)
                temp_path_vars.append(other_var)
                current_conditioning = found_edge.conditioning_variables
                
            if success:
                chosen_bottom = current_bottom
                path_edges = temp_path_edges
                path_vars = temp_path_vars[::-1] # Reverse to match row 0 up to t_top
                break
                
        if chosen_bottom is None:
            raise ValueError(f"Proximity condition violation or disconnected paths at column {e}.")
            
        # Map path variables to the matrix columns
        for t, var in enumerate(path_vars):
            matrix[t, e] = var
            
        # Place the pivot element on the pseudo-diagonal boundary
        matrix[d - 1 - e, e] = chosen_bottom
        placed_diagonals.add(chosen_bottom)
        
        # Map copulas to the pair_copulas layout
        for t, edge in enumerate(path_edges[::-1]):
            pair_copulas[t][e] = edge.copula
            used_edges.add(edge)
            
    # Solve for the final remaining diagonal variable in the top-right spot
    all_vars = set(range(1, d + 1))
    remaining_vars = all_vars - placed_diagonals
    if len(remaining_vars) == 1:
        matrix[0, d - 1] = list(remaining_vars)[0]
    else:
        # Fallback if your custom nodes do not perfectly span 1 to d
        all_unique_nodes = set().union(*(edge.conditioned_variables for edge in custom_trees[0]))
        matrix[0, d - 1] = list(all_unique_nodes - placed_diagonals)[0]

    
    # Reset indexing on all upper triangles to 1-based for pyvinecopulib compatibility
    for i in range(d):
        for j in range(d-i):
            matrix[i,j] +=1

    return matrix, pair_copulas




# --- OLD FUNCTIONS (deprecated) -------------------------------------------------------------
def tail_weighted_kendall_tau(
    u: np.ndarray,
    v: np.ndarray,
    q: float = 0.2,
    tail: str = "both",
    min_obs: int = 10,
) -> float:
    """
    Compute Kendall's tau restricted to the tail region.
 
    Parameters
    ----------
    u, v : np.ndarray, shape (n,)
        Pseudo-observations for a single pair, values in (0, 1).
    q : float
        Tail quantile threshold. Observations with u < q AND v < q
        (lower tail) or u > 1-q AND v > 1-q (upper tail) are retained.
    tail : str
        "lower"  — restrict to joint lower tail (u < q, v < q)
        "upper"  — restrict to joint upper tail (u > 1-q, v > 1-q)
        "both"   — union of lower and upper tail (default)
    min_obs : int
        Minimum number of tail observations required to compute tau.
        Returns 0.0 if the tail region is too sparse (avoids noise).
 
    Returns
    -------
    float
        Tail-weighted Kendall's tau. Returns 0.0 if insufficient tail obs.
    """
    if tail == "lower":
        mask = (u < q) & (v < q)
    elif tail == "upper":
        mask = (u > 1 - q) & (v > 1 - q)
    elif tail == "both":
        mask_lower = (u < q) & (v < q)
        mask_upper = (u > 1 - q) & (v > 1 - q)
        mask = mask_lower | mask_upper
    else:
        raise ValueError(f"tail must be 'lower', 'upper', or 'both', got '{tail}'")
 
    u_tail, v_tail = u[mask], v[mask]
 
    if len(u_tail) < min_obs:
        warnings.warn(
            f"Only {len(u_tail)} tail observations (threshold q={q}). "
            f"Returning 0.0 for this pair. Consider increasing q or n.",
            RuntimeWarning,
        )
        return 0.0
 
    tau, _ = kendalltau(u_tail, v_tail)
    return float(tau) if np.isfinite(tau) else 0.0
 
 
def pairwise_tail_tau_matrix(
    pseudo_obs: np.ndarray,
    q: float = 0.2,
    tail: str = "both",
    min_obs: int = 10,
) -> np.ndarray:
    """
    Compute the d x d matrix of pairwise tail-weighted Kendall's tau.
 
    Parameters
    ----------
    pseudo_obs : np.ndarray, shape (n, d)
    q, tail, min_obs : passed to tail_weighted_kendall_tau()
 
    Returns
    -------
    np.ndarray, shape (d, d)
        Symmetric matrix with zeros on the diagonal.
    """
    n, d = pseudo_obs.shape
    tau_matrix = np.zeros((d, d))
 
    for i, j in combinations(range(d), 2):
        tw_tau = tail_weighted_kendall_tau(
            pseudo_obs[:, i], pseudo_obs[:, j],
            q=q, tail=tail, min_obs=min_obs,
        )
        tau_matrix[i, j] = tw_tau
        tau_matrix[j, i] = tw_tau
 
    return tau_matrix
 
 
def maximum_spanning_tree_edges(weight_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    Compute the Maximum Spanning Tree (MST) of a complete graph with given
    edge weights using scipy (which computes minimum spanning tree, so we
    negate the weights).
 
    Parameters
    ----------
    weight_matrix : np.ndarray, shape (d, d)
        Symmetric matrix of edge weights (e.g. absolute tau values).
 
    Returns
    -------
    list of (i, j) tuples — edges of the MST, 0-indexed.
    """
    # Use absolute values: MST should be on |tau| to handle negative dependence
    abs_weights = np.abs(weight_matrix)
 
    # Negate for minimum_spanning_tree (scipy finds minimum, we want maximum)
    neg_weights = -abs_weights
    np.fill_diagonal(neg_weights, 0.0)
 
    sparse = csr_matrix(np.triu(neg_weights, k=1))
    mst = minimum_spanning_tree(sparse)
    mst_coo = mst.tocoo()
 
    edges = [(int(i), int(j)) for i, j in zip(mst_coo.row, mst_coo.col)]
    return edges


def build_rvine_structure_from_tree1(
    d: int,
    tree1_edges: list[tuple[int, int]],
) -> pvc.RVineStructure:
    """
    Construct an RVineStructure whose Tree 1 matches the provided edges.
    Remaining trees are completed via the default sequential construction
    (Dissmann), which pyvinecopulib handles internally when
    fitting a Vinecop with a fixed structure.
 
    The approach: encode Tree 1 as an order + triangular matrix, then
    pass to RVineStructure. pyvinecopulib accepts an explicit structure
    matrix (d x d upper triangular).
 
    Parameters
    ----------
    d : int
        Number of variables.
    tree1_edges : list of (i, j) tuples
        Edges of Tree 1, 0-indexed.
 
    Returns
    -------
    pvc.RVineStructure (D-Vine limitation!)
    """
    # Build adjacency from Tree 1 edges to extract a root-based order.
    # Use a simple DFS to get the variable ordering from the MST.
    from collections import defaultdict, deque
 
    adj = defaultdict(list)
    for i, j in tree1_edges:
        adj[i].append(j)
        adj[j].append(i)
 
    # BFS from node with highest degree (natural root for vine ordering)
    degrees = {node: len(neighbours) for node, neighbours in adj.items()}
    root = max(degrees, key=degrees.get)
 
    visited = []
    queue = deque([root])
    seen = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        for nb in sorted(adj[node], key=lambda x: -degrees[x]):
            if nb not in seen:
                queue.append(nb)
 
    order = visited  # length d, 0-indexed
 
    # pyvinecopulib's RVineStructure can be initialised from an order array
    # (1-indexed). This sets Tree 1 to the path defined by the order, which
    # is a valid D-vine. For the full MST topology (non-path trees), we need
    # the full structure matrix.
    #
    # Strategy: use the order-based construction as an approximation when the
    # MST is a path, otherwise fall back to a D-vine on the BFS order and
    # note the limitation.
 
    order_1indexed = np.array([o + 1 for o in order], dtype=np.uint64)
 
    try:
        struct = pvc.RVineStructure.from_order(order_1indexed)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build RVineStructure from MST order: {e}\n"
            f"Order attempted: {order_1indexed}"
        )
 
    return struct