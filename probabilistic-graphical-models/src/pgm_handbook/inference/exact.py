"""Exact inference by enumeration and variable elimination."""

from __future__ import annotations

import itertools
from typing import Dict, List, Mapping, Sequence, Tuple

from ..representation.bayesian_network import BayesianNetwork


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def enumeration_ask(
    query: str,
    evidence: Mapping[str, object],
    bn: BayesianNetwork,
) -> Dict:
    """``P(query | evidence)`` by enumerating hidden variables."""
    domains = bn.domains()
    hidden = [v for v in bn.nodes() if v != query and v not in evidence]
    result: Dict = {}
    for qv in domains[query]:
        if query in evidence and qv != evidence[query]:
            result[qv] = 0.0
            continue
        total = 0.0
        for combo in itertools.product(*(domains[v] for v in hidden)):
            assignment = dict(evidence)
            assignment[query] = qv
            assignment.update(dict(zip(hidden, combo)))
            total += bn.joint_probability(assignment)
        result[qv] = total
    Z = sum(result.values())
    if Z == 0.0:
        raise ValueError("Evidence has zero probability")
    return {k: v / Z for k, v in result.items()}


# ---------------------------------------------------------------------------
# Variable elimination
# ---------------------------------------------------------------------------
class _Factor:
    """A factor over ``scope`` with a table of probabilities."""

    def __init__(self, scope: Tuple[str, ...], table: Dict, domains: Dict[str, List]):
        self.scope = scope
        self.table = table
        self.domains = domains

    def __mul__(self, other: "_Factor") -> "_Factor":
        ordered = list(self.scope)
        for v in other.scope:
            if v not in ordered:
                ordered.append(v)
        new_scope = tuple(ordered)
        new_table: Dict = {}
        for combo in _all_combos(new_scope, self.domains):
            assignment = dict(zip(new_scope, combo))
            k_self = tuple(assignment[v] for v in self.scope)
            k_other = tuple(assignment[v] for v in other.scope)
            if k_self in self.table and k_other in other.table:
                new_table[combo] = self.table[k_self] * other.table[k_other]
        return _Factor(new_scope, new_table, self.domains)

    def sum_out(self, var: str) -> "_Factor":
        new_scope = tuple(v for v in self.scope if v != var)
        new_table: Dict = {}
        for combo in _all_combos(new_scope, self.domains):
            assignment = dict(zip(new_scope, combo))
            total = 0.0
            for val in self.domains[var]:
                full = tuple(
                    assignment[v] if v != var else val for v in self.scope
                )
                if full in self.table:
                    total += self.table[full]
            new_table[combo] = total
        return _Factor(new_scope, new_table, self.domains)


def _all_combos(scope: Sequence[str], domains: Mapping[str, Sequence]):
    if not scope:
        return [()]
    return list(itertools.product(*(domains[v] for v in scope)))


def variable_elimination(
    query: str,
    evidence: Mapping[str, object],
    bn: BayesianNetwork,
    elimination_order: Sequence[str] | None = None,
) -> Dict:
    """``P(query | evidence)`` via variable elimination."""
    domains = bn.domains()

    # Build CPD factors.
    factors: List[_Factor] = []
    for v in bn.nodes():
        cpd = bn.get_cpd(v)
        scope = (v,) + cpd.parents
        table: Dict = {}
        for combo in _all_combos(scope, domains):
            assignment = dict(zip(scope, combo))
            if any(
                k in evidence and evidence[k] != assignment[k]
                for k in scope
            ):
                continue
            table[combo] = cpd.prob(
                assignment[v], tuple(assignment[p] for p in cpd.parents)
            )
        factors.append(_Factor(scope, table, domains))

    # Determine elimination order.
    # Keep the network's node order so runs are deterministic.
    hidden = [v for v in bn.nodes() if v != query and v not in evidence]
    if elimination_order is None:
        order = list(hidden)
    else:
        order = [v for v in elimination_order if v in hidden]
        for v in hidden:
            if v not in order:
                order.append(v)

    # Eliminate one variable at a time.
    for var in order:
        involved = [f for f in factors if var in f.scope]
        rest = [f for f in factors if var not in f.scope]
        if not involved:
            continue
        combined = involved[0]
        for f in involved[1:]:
            combined = combined * f
        factors = rest + [combined.sum_out(var)]

    # Multiply remaining factors.
    result = factors[0]
    for f in factors[1:]:
        result = result * f

    # Extract the posterior over ``query``.
    dist: Dict = {v: 0.0 for v in domains[query]}
    for combo, val in result.table.items():
        assignment = dict(zip(result.scope, combo))
        if any(
            k in evidence and assignment[k] != evidence[k]
            for k in result.scope
        ):
            continue
        qv = assignment[query]
        dist[qv] = dist.get(qv, 0.0) + val

    Z = sum(dist.values())
    if Z == 0.0:
        raise ValueError("Evidence has zero probability")
    return {k: v / Z for k, v in dist.items()}
