import numpy as np
import pyvinecopulib as pvc
from typing import Self

from tools.Helpers import _decode_edge


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
        Extracts the vine structure into a list of edge sets, one set per tree level.
        Each edge is represented as (conditioned_pair, conditioning_set).
        """
        structure = vine.structure
        edge_sets = cls._extract_edge_sets(structure)
        n_vars = structure.dim

        return cls(edge_sets=edge_sets, n_vars=n_vars)

    @classmethod
    def _extract_edge_sets(cls, structure: pvc.RVineStructure) -> list[set[tuple[frozenset[int], frozenset[int]]]]:
        """
        Extract the canonical edge objects at each tree level.

        The returned representation is a set of tuples
        (conditioned_pair, conditioning_set) for each tree.
        """
        mat = structure.matrix
        d = structure.dim
        edge_sets = []

        for tree in range(d - 1):
            edges: set[tuple[frozenset[int], frozenset[int]]] = set()
            for edge in range(d - tree - 1):
                edges.add(_decode_edge(mat, d, tree, edge))
            edge_sets.append(edges)

        return edge_sets


    def get_edges(self):
        """
        Returns a list of sets of tuples (conditioned_pair, conditioning_set). The list is indexed by tree level, and each set contains the edges for that tree.
        """
        zero_indexed_edge_sets = []

        for tree_edges in self.edge_sets:
            zero_indexed_tree_edges = set()
            for conditioned_pair, conditioning_set in tree_edges:
                zero_indexed_tree_edges.add(
                    (
                        frozenset(int(label) - 1 for label in conditioned_pair),
                        frozenset(int(label) - 1 for label in conditioning_set),
                    )
                )
            zero_indexed_edge_sets.append(zero_indexed_tree_edges)

        return zero_indexed_edge_sets


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
            weights = np.array([0.5 ** k for k in range(n_trees)])
            weights /= weights.sum()
        else:
            weights = tree_weights

        assert weights is not None

        total_distance = 0.0

        for k in range(n_trees):
            edges_a = self.edge_sets[k]
            edges_b = other.edge_sets[k]

            if not edges_a and not edges_b:
                continue

            intersection = len(edges_a & edges_b)
            union = len(edges_a | edges_b)
            jaccard_dist = 1.0 - (intersection / union) if union > 0 else 0.0
            total_distance += weights[k] * jaccard_dist

        return total_distance