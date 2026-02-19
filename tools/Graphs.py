import networkx as nx
from typing import Any, Optional
from numpy.typing import NDArray
import numpy as np


def get_name(vc:Any, tree:int, edge:int, vars_names:list[str]):
    """
    Helper:Retrieves the name of an graph object in a VineCopula structure, either a node or an edge, e.g. 1,3|2 for the 
    copula between variables 1 and 3 conditioned on 2.

    :param vc: pyvinecopulibVineCopula object
    :param tree: int, for which tree to extract the name
    :param edge: int, for which edge in tree to extract the name
    :param vars_names: list[str], the variable names

    :return: str, a object name, e.g. "1,2|3" for the copula between variables 1 and 2 conditioned on 3
    """
    M = vc.matrix
    d = M.shape[0]

    # Conditioned set
    bef_indices = [d - edge - 1, tree]  # Adjusted for zero-indexing
    bef = ",".join([vars_names[int(M[i, edge]) - 1] for i in bef_indices])

    # Conditioning set
    if tree > 0:
        aft = ",".join([vars_names[int(M[i - 1, edge]) - 1] for i in range(tree, 0, -1)])
    else:
        aft = ""

    # Separator
    sep = "|" if tree > 0 else ""

    # Combine everything
    return bef + sep + aft


def get_graph(tree: int, vc: Any, vars_names: list[str]) -> tuple[NDArray[np.int_], dict[int, str], dict[tuple[int, int], str]]:
    """
    Helper: Extracts the adjuctancy matrix to build the graph as well as node and edge labels.

    :param tree: int, for which tree to extract the name
    :param vc: pyvinecopulibVineCopula object
    :param vars_names: list[str], the variable names

    :return: tuple, adjacency matrix to build graph with nx.from_adjacency_matrix, the node labels, and edge labels
    """
    
    M = vc.matrix
    d = vc.dim

    adj_mat = np.zeros((d - tree, d - tree), dtype=int)

  # Extract node and edge labels as numbers
    if tree > 0:
        vertices = edges = np.zeros((d - tree, tree + 1), dtype=int)
        for j in range(d - tree):
            rows = np.array([d - j - 1, *range(tree - 1, -1, -1)])
            vertices[j, :] = M[rows, j]
    else:
        vertices = np.diag(M[d - 1 :: -1, :]).reshape(-1, 1)

    edges = np.zeros((d - tree - 1, tree + 2), dtype=int)
    for j in range(d - tree - 1):
        rows = np.array([d - j - 1, *range(tree, -1, -1)])
        edges[j, :] = M[rows, j]

    # Build adjacency matrix by matching vertices and edges
    edge_labels = {}

    for i in range(edges.shape[0]):
        ind_i = []

        for j in range(vertices.shape[0]):
            if np.all(np.isin(vertices[j, :], edges[i, :])):
                ind_i.append(j)

        adj_mat[ind_i[0], ind_i[1]] = 1
        adj_mat[ind_i[1], ind_i[0]] = 1
        edge_labels[(ind_i[0], ind_i[1])] = get_name(vc, tree, i, vars_names)

    # Node labels
    if tree > 0:
        node_labels = {i: get_name(vc, tree - 1, i, vars_names) for i in range(d - tree)}
    else:
        node_labels = {j: vars_names[int(M[d - j - 1, j]) - 1] for j in range(d)}

    return adj_mat, node_labels, edge_labels


def make_graph_network(vc:Any, trees:Optional[list[int]|int]=None, vars_names:Optional[list[str]]=None):
    """
    Builds a graph of distinct trees for a VineCopula. 

    :param vc: pyvinecopulibVineCopula object
    :param trees: Either for a subset of trees, e.g. [0,1,2] or for all trees.
    :param vars_names: A list of variable names, e.g. from portfolio_composition

    :return: A networkx graph of distinct networks
    """
    if not trees:
        trees = list(range(vc.trunc_lvl))
    if isinstance(trees,int):
        trees  = [trees]

    if vars_names is not None:
        if len(vars_names) != vc.dim:
            raise ValueError("The number of variable names must be equal to the dimension of the vine copula.")
    else:
        vars_names = [str(i) for i in range(vc.dim)]

    graphs = []
    fams = []

    for tree in trees:
        adj_mat, node_labels, edge_labels = get_graph(tree, vc, vars_names)

        g = nx.from_numpy_array(adj_mat)
            
        for node, label in node_labels.items():
            _family = fams[node] if tree>0 else "asset" # from previous iteration

            g.nodes[node]["title"] = label+"\n"+_family+"\n"+f"tree {tree}" # Controls hover label in pyvis, can be html
            g.nodes[node]["tree"] = tree
            g.nodes[node]["label"] = label # Label in pyvis
            g.nodes[node]["shape"] = "box" if tree == 0 else "ellipse" # Controls shape in pyvis
            g.nodes[node]["group"] = f"Tree {tree}"  # Controls color in pyvis
            g.nodes[node]["family"] = _family   

        _fams = []
        for i, (edge, label) in enumerate(edge_labels.items()):
            _family = vc.get_pair_copula(tree,i).family.name
            
            g.edges[edge]["title"] = label+"\n"+_family+"\n"+f"tree {tree}"
            g.edges[edge]["tree"] = tree
            g.edges[edge]["label"] = label+"\n"+_family
            g.edges[edge]["group"] = f"Tree {tree}" 
            g.edges[edge]["family"] = _family

            _fams.append(_family)
        
        fams = _fams
        graphs.append(g)

    G = nx.disjoint_union_all(graphs)

    return G


def get_node_labels(G:nx.Graph):
    """
    Returns a dictionary with node labels.
    """
    return {node: G.nodes[node]["label"] for node in G.nodes}

def get_edge_labels(G:nx.Graph):
    """
    Returns a dictionary with edge labels.
    """
    return {edge: G.edges[edge]["label"] for edge in G.edges}