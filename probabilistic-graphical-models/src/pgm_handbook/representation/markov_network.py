r"""Markov (undirected) networks with clique potentials and partition function.

The joint distribution is

    P(x) = (1/Z) \prod_{c \in C} \phi_c(x_c),    Z = \sum_x \prod_c \phi_c(x_c).

Each ``add_factor`` call registers an *unnormalised* potential ``phi_c``.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx


class MarkovNetwork:
    """Undirected graphical model with tabular clique potentials."""

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._factors: List[Tuple[Tuple[str, ...], Dict[Tuple, float]]] = []

    # ------------------------------------------------------------------
    # Structure & potentials
    # ------------------------------------------------------------------
    def add_node(self, name: str) -> None:
        self._graph.add_node(name)

    def add_edge(self, u: str, v: str) -> None:
        self._graph.add_edge(u, v)

    def add_factor(
        self,
        scope: Sequence[str],
        table: Mapping[Tuple, float],
    ) -> None:
        """Attach a clique potential ``phi_c`` with the given scope and table."""
        scope = tuple(scope)
        for v in scope:
            self.add_node(v)
        for a, b in itertools.combinations(scope, 2):
            self.add_edge(a, b)
        self._factors.append((scope, {tuple(k): float(v) for k, v in table.items()}))

    # ------------------------------------------------------------------
    # Cliques & probabilities
    # ------------------------------------------------------------------
    def maximal_cliques(self) -> List[Tuple[str, ...]]:
        """Return all maximal cliques of the undirected graph."""
        return [tuple(sorted(c)) for c in nx.find_cliques(self._graph)]

    def unnormalized_probability(self, assignment: Mapping[str, object]) -> float:
        r"""Return ``\prod_c \phi_c(x_c)`` for a full assignment."""
        prod = 1.0
        for scope, table in self._factors:
            key = tuple(assignment[v] for v in scope)
            prod *= table.get(key, 0.0)
        return prod

    def log_linear_energy(
        self,
        assignment: Mapping[str, object],
        weights: Sequence[Tuple[Sequence[str], Mapping[Tuple, float]]],
    ) -> float:
        r"""Return a log-linear energy ``\sum_c w_c \cdot f_c(x_c)``.

        Parameters
        ----------
        assignment : mapping
        weights : sequence of ``(scope, table)`` pairs
            ``table`` maps a joint instantiation to its potential in log-space.
        """
        energy = 0.0
        for scope, table in weights:
            key = tuple(assignment[v] for v in scope)
            energy += table.get(key, -math.inf)
        return energy

    # ------------------------------------------------------------------
    # Partition function
    # ------------------------------------------------------------------
    def partition_function(self, domains: Mapping[str, Sequence]) -> float:
        """Exact ``Z`` by enumeration over the Cartesian product of domains."""
        nodes = list(self._graph.nodes)
        Z = 0.0
        for combo in itertools.product(*(domains[v] for v in nodes)):
            assignment = dict(zip(nodes, combo))
            Z += self.unnormalized_probability(assignment)
        return Z

    def approximate_partition_function(
        self,
        proposal_sample,
        proposal_unnorm,
        n_samples: int = 10_000,
        rng=None,
    ) -> float:
        """Importance-sampling estimate of ``Z``.

        ``Z ≈ mean( phi~(x_i) / q(x_i) )`` with ``x_i ~ q``.  ``proposal_unnorm``
        must be the *normalised* proposal density ``q`` for the result to be
        ``Z``; ``proposal_sample(rng)`` must return an assignment dict.
        """
        import numpy as np

        rng = rng or np.random.default_rng()
        weights = []
        for _ in range(n_samples):
            x = proposal_sample(rng)
            weights.append(self.unnormalized_probability(x) / proposal_unnorm(x))
        return float(np.mean(weights))
