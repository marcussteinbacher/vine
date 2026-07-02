import numpy as np
import pyvinecopulib as pvc
from typing import Self


class VineStructureSnapshot:
    def __init__(self, edge_sets,  n_vars):
        """
        Initialize a VineStructureSnapshot.
        """
        self.edge_sets = edge_sets
        self.n_vars = n_vars

    @classmethod
    def from_vinecop(cls, vine: pvc.Vinecop):
        """
        Build a VineStructureSnapshot from a pyvinecopulib.Vinecop object.
        Extracts the structure matrix and converts it into a list of edge sets for each tree level.
        Each edge is represented as a frozenset of the conditioned pair and a frozenset of the conditioning set.
        """
        structure = vine.structure
        mat = structure.matrix
        edge_sets = cls._extract_edge_sets(structure)
        n_vars = mat.shape[0]

        return cls(edge_sets=edge_sets, n_vars=n_vars)

    @classmethod
    def _extract_edge_sets(cls, structure: pvc.RVineStructure) -> list[set]:
        """
        Extract edges at each tree level from the vine structure matrix.
        An edge at tree k is a pair (conditioned set) given a conditioning set.
        Represented as frozenset for order-invariance.
        """
        mat = structure.matrix  # shape (d, d)
        d = mat.shape[0]
        edge_sets = []

        for tree in range(d - 1):  # tree levels 0..d-2
            edges = set()
            for col in range(tree + 1, d):
                # conditioned pair
                node_a = mat[tree, col]
                node_b = mat[col, col]       # diagonal = variable label
                # conditioning set: column above tree level
                cond = frozenset(mat[tree+1:col, col])
                edge = (frozenset([node_a, node_b]), cond)
                edges.add(edge)
            edge_sets.append(edges)

        return edge_sets


    def weighted_jaccard_distance(self, other:Self, tree_weights: np.ndarray | None = None) -> float:
        """
        Weighted Jaccard distance across tree levels.
        Tree 1 gets the highest weight by default (geometric decay).
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        d = self.n_vars
        n_trees = d - 1

        if tree_weights is None:
            # Exponential decay: tree 1 has weight 1, tree 2 has 0.5, etc.
            tree_weights = np.array([0.5 ** k for k in range(n_trees)])
            tree_weights /= tree_weights.sum()

        total_distance = 0.0

        for k in range(n_trees):
            edges_a = self.edge_sets[k]
            edges_b = other.edge_sets[k]

            if not edges_a and not edges_b:
                continue

            intersection = len(edges_a & edges_b)
            union = len(edges_a | edges_b)
            jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0.0
            total_distance += tree_weights[k] * jaccard_dist

        return total_distance