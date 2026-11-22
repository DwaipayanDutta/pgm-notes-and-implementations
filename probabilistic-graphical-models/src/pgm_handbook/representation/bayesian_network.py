r"""Bayesian networks: DAG structure, CPDs, joint factorisation and d-separation.

A Bayesian network factorises a joint distribution as

    P(X_1, ..., X_n) = \prod_i P(X_i | Pa(X_i)).

Each CPD provides ``prob(value, parent_assignment)`` and ``domain``.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx


class TabularCPD:
    """A conditional probability table.

    Parameters
    ----------
    variable : str
        The child variable.
    parents : sequence of str
        Parent variables (order matches keys of ``table``).
    table : mapping of tuple -> mapping
        ``table[parent_assignment][value] = P(value | parents)``.
        For root variables use the key ``()``.
    """

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        table: Mapping[Tuple, Mapping[str, float]],
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.table: Dict[Tuple, Dict[str, float]] = {
            tuple(k): dict(v) for k, v in table.items()
        }
        domain: Set = set()
        for dist in self.table.values():
            domain.update(dist.keys())
        self.domain: List = sorted(domain, key=str)

    def distribution(self, parent_assignment: Sequence) -> Dict:
        """Return ``P(variable | parent_assignment)`` as a dict."""
        return dict(self.table.get(tuple(parent_assignment), {}))

    def prob(self, value, parent_assignment: Sequence) -> float:
        """Return ``P(variable = value | parent_assignment)``."""
        return self.table.get(tuple(parent_assignment), {}).get(value, 0.0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TabularCPD({self.variable!r}, parents={self.parents})"


class BayesianNetwork:
    """Directed acyclic graphical model with tabular or local CPDs."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._cpds: Dict[str, "CPDLike"] = {}

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    def add_node(self, name: str) -> None:
        """Add ``name`` to the DAG (idempotent)."""
        self._graph.add_node(name)

    def add_edge(self, parent: str, child: str) -> None:
        """Add ``parent -> child`` after checking that no cycle is created."""
        if parent not in self._graph:
            raise ValueError(f"Unknown parent node {parent!r}")
        if child not in self._graph:
            raise ValueError(f"Unknown child node {child!r}")
        if parent == child:
            raise ValueError("Self-loops are not allowed")
        if nx.has_path(self._graph, child, parent):
            raise ValueError(
                f"Adding edge {parent}->{child} would create a cycle"
            )
        self._graph.add_edge(parent, child)

    def nodes(self) -> List[str]:
        """Return all node names (insertion order)."""
        return list(self._graph.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        """Return all directed edges."""
        return list(self._graph.edges)

    def parents(self, node: str) -> List[str]:
        """Return the parents of ``node`` in insertion order."""
        return list(self._graph.predecessors(node))

    # ------------------------------------------------------------------
    # CPDs & probability
    # ------------------------------------------------------------------
    def set_cpd(self, cpd: "CPDLike") -> None:
        """Attach ``cpd`` to its variable and register its parents as edges."""
        if cpd.variable not in self._graph:
            self._graph.add_node(cpd.variable)
        for p in cpd.parents:
            if p not in self._graph:
                self._graph.add_node(p)
            if not self._graph.has_edge(p, cpd.variable):
                self.add_edge(p, cpd.variable)
        self._cpds[cpd.variable] = cpd

    def get_cpd(self, node: str) -> "CPDLike":
        """Return the CPD attached to ``node``."""
        return self._cpds[node]

    def joint_probability(self, assignment: Mapping[str, object]) -> float:
        """Return ``P(X_1, ..., X_n)`` for a full assignment."""
        prob = 1.0
        for var in self._graph.nodes:
            cpd = self._cpds[var]
            parent_vals = tuple(assignment[p] for p in cpd.parents)
            prob *= cpd.prob(assignment[var], parent_vals)
        return prob

    def domain(self, node: str) -> List:
        """Return the discrete domain of ``node``."""
        return list(self._cpds[node].domain)

    def domains(self) -> Dict[str, List]:
        """Return a mapping node -> list of possible values."""
        return {v: self.domain(v) for v in self._graph.nodes}

    # ------------------------------------------------------------------
    # d-separation
    # ------------------------------------------------------------------
    def is_d_separated(
        self,
        X: Sequence[str],
        Y: Sequence[str],
        Z: Sequence[str],
    ) -> bool:
        """Return True iff ``X`` and ``Y`` are d-separated given ``Z``.

        Uses the moralised ancestral-graph criterion (Koller & Friedman,
        Algorithm 3.1).
        """
        X, Y, Z = set(X), set(Y), set(Z)
        ancestors = self._ancestors(X | Y | Z)

        # 1. Build the ancestral graph.
        g = nx.Graph()
        for v in ancestors:
            g.add_node(v)
        for v in ancestors:
            for pa in self._graph.predecessors(v):
                if pa in ancestors:
                    g.add_edge(pa, v)

        # 2. Moralise: connect all co-parents.
        for v in ancestors:
            pas = [p for p in self._graph.predecessors(v) if p in ancestors]
            for a, b in itertools.combinations(pas, 2):
                g.add_edge(a, b)

        # 3. Remove evidence.
        g.remove_nodes_from(Z)

        # 4. Check connectivity.
        for x in X & ancestors:
            for y in Y & ancestors:
                if nx.has_path(g, x, y):
                    return False
        return True

    def _ancestors(self, nodes: Set[str]) -> Set[str]:
        """Return the union of ancestors of every node in ``nodes``."""
        result: Set[str] = set(nodes)
        stack = list(nodes)
        while stack:
            n = stack.pop()
            for pa in self._graph.predecessors(n):
                if pa not in result:
                    result.add(pa)
                    stack.append(pa)
        return result


CPDLike = object  # duck-typed: .variable, .parents, .prob, .domain
