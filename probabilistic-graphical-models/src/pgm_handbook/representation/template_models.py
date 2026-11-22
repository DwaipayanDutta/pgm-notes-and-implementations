"""Plate models and (2-slice) dynamic Bayesian networks."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .bayesian_network import BayesianNetwork, TabularCPD


class PlateModel:
    """A plate model: a template replicated over an index.

    Each template entry is ``(name_template, parent_templates, cpd_builder)``.
    Both ``name_template`` and ``parent_templates`` are ``str.format`` templates
    using ``{i}`` for the plate index.  ``cpd_builder(var_name, parent_names)``
    must return a CPD (usually a :class:`TabularCPD`).
    """

    def __init__(self, index_name: str, index_values: Sequence) -> None:
        self.index_name = index_name
        self.index_values: List = list(index_values)
        self._templates: List[
            Tuple[str, Tuple[str, ...], Callable[[str, List[str]], TabularCPD]]
        ] = []

    def add_template(
        self,
        name_template: str,
        parent_templates: Sequence[str],
        cpd_builder: Callable[[str, List[str]], TabularCPD],
    ) -> None:
        self._templates.append(
            (name_template, tuple(parent_templates), cpd_builder)
        )

    def ground(self) -> BayesianNetwork:
        """Ground all templates into a concrete :class:`BayesianNetwork`."""
        bn = BayesianNetwork()

        # 1. Add every grounded node.
        for i in self.index_values:
            for name_t, _, _ in self._templates:
                bn.add_node(name_t.format(i=i))

        # 2. Add edges.
        for i in self.index_values:
            for name_t, parents_t, _ in self._templates:
                child = name_t.format(i=i)
                for pt in parents_t:
                    parent = pt.format(i=i)
                    bn.add_edge(parent, child)

        # 3. Attach CPDs.
        for i in self.index_values:
            for name_t, parents_t, builder in self._templates:
                child = name_t.format(i=i)
                parent_names = [pt.format(i=i) for pt in parents_t]
                bn.set_cpd(builder(child, parent_names))
        return bn


def _rename_cpd(cpd: TabularCPD, new_var: str, parent_mapper) -> TabularCPD:
    """Return a copy of ``cpd`` with renamed variable and parents."""
    new_parents = tuple(parent_mapper(p) for p in cpd.parents)
    new_table: Dict = {}
    for pa_key, dist in cpd.table.items():
        new_table[tuple(pa_key)] = dict(dist)
    return TabularCPD(new_var, new_parents, new_table)


class DynamicBayesianNetwork:
    """2-slice temporal model unrollable across ``T`` time steps.

    * ``set_initial`` — CPD for slice 0.
    * ``set_transition(cpd)`` / ``set_transition(var, cpd)`` — CPD at ``t`` whose parents lie in slice
      ``t-1`` (their names are automatically remapped).
    * ``add_observation(var, cpd)`` — observation CPD whose parent is a state
      variable in the same slice.
    """

    def __init__(self, state_vars: Sequence[str]) -> None:
        self.state_vars: List[str] = list(state_vars)
        self.obs_vars: List[str] = []
        self.initial_cpds: Dict[str, TabularCPD] = {}
        self.transition_cpds: Dict[str, TabularCPD] = {}
        self.observation_cpds: Dict[str, TabularCPD] = {}

    def set_initial(self, cpd: TabularCPD) -> None:
        self.initial_cpds[cpd.variable] = cpd

    def set_transition(self, var_or_cpd, cpd: TabularCPD | None = None) -> None:
        """Register a transition CPD: ``set_transition(cpd)`` or ``set_transition(var, cpd)``."""
        if cpd is None:
            cpd = var_or_cpd
            var = cpd.variable
        else:
            var = var_or_cpd
        self.transition_cpds[var] = cpd

    def add_observation(self, var: str, cpd: TabularCPD) -> None:
        self.obs_vars.append(var)
        self.observation_cpds[var] = cpd

    def unroll(self, T: int) -> BayesianNetwork:
        """Return a Bayesian network with ``T`` time slices."""
        if T < 1:
            raise ValueError("T must be >= 1")
        bn = BayesianNetwork()
        for t in range(T):
            for v in self.state_vars:
                bn.add_node(f"{v}_{t}")
            for o in self.obs_vars:
                bn.add_node(f"{o}_{t}")

        # Initial slice (t = 0)
        for v in self.state_vars:
            cpd = _rename_cpd(self.initial_cpds[v], f"{v}_0", lambda p: f"{p}_0")
            bn.set_cpd(cpd)

        # Transition slices (t >= 1)
        for t in range(1, T):
            for v in self.state_vars:
                cpd = self.transition_cpds[v]
                new = _rename_cpd(cpd, f"{v}_{t}", lambda p: f"{p}_{t-1}")
                bn.set_cpd(new)

        # Observations at every slice
        for t in range(T):
            for o in self.obs_vars:
                cpd = self.observation_cpds[o]
                new = _rename_cpd(cpd, f"{o}_{t}", lambda p: f"{p}_{t}")
                bn.set_cpd(new)

        return bn
