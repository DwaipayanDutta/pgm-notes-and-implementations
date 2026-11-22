"""Tests for Bayesian networks, Markov networks, plate models, DBNs, local CPDs."""

from __future__ import annotations

import math

import pytest

from pgm_handbook.representation.bayesian_network import (
    BayesianNetwork,
    TabularCPD,
)
from pgm_handbook.representation.markov_network import MarkovNetwork
from pgm_handbook.representation.local_structures import (
    DeterministicCPD,
    MultiplexerCPD,
    NoisyORCPD,
    NoisyMAXCPD,
    LogisticCPD,
    LinearGaussianCPD,
)
from pgm_handbook.representation.template_models import (
    PlateModel,
    DynamicBayesianNetwork,
)


# ---------------------------------------------------------------------------
# BayesianNetwork
# ---------------------------------------------------------------------------
def _build_simple_bn() -> BayesianNetwork:
    bn = BayesianNetwork()
    bn.add_node("A"); bn.add_node("B"); bn.add_node("C")
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.set_cpd(
        TabularCPD("A", (), {(): {"T": 0.3, "F": 0.7}})
    )
    bn.set_cpd(
        TabularCPD(
            "B", ("A",),
            {("T",): {"T": 0.9, "F": 0.1},
             ("F",): {"T": 0.2, "F": 0.8}},
        )
    )
    bn.set_cpd(
        TabularCPD(
            "C", ("B",),
            {("T",): {"T": 0.8, "F": 0.2},
             ("F",): {"T": 0.1, "F": 0.9}},
        )
    )
    return bn


def test_cycle_detection():
    bn = BayesianNetwork()
    bn.add_node("A"); bn.add_node("B"); bn.add_node("C")
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    with pytest.raises(ValueError):
        bn.add_edge("C", "A")


def test_self_loop_rejected():
    bn = BayesianNetwork()
    bn.add_node("A")
    with pytest.raises(ValueError):
        bn.add_edge("A", "A")


def test_joint_probability_matches_factorisation():
    bn = _build_simple_bn()
    p = bn.joint_probability({"A": "T", "B": "T", "C": "F"})
    expected = 0.3 * 0.9 * 0.2
    assert math.isclose(p, expected, rel_tol=1e-12)


def test_joint_distribution_normalises():
    bn = _build_simple_bn()
    total = 0.0
    for a in ["T", "F"]:
        for b in ["T", "F"]:
            for c in ["T", "F"]:
                total += bn.joint_probability({"A": a, "B": b, "C": c})
    assert math.isclose(total, 1.0, rel_tol=1e-12)


def test_d_separation_chain():
    # A -> B -> C.  A ⟂ C | B, but A NOT ⟂ C | ∅.
    bn = _build_simple_bn()
    assert bn.is_d_separated(["A"], ["C"], ["B"])
    assert not bn.is_d_separated(["A"], ["C"], [])


def test_d_separation_v_structure():
    # Classic v-structure: A -> C <- B.  A ⟂ B | ∅, A NOT ⟂ B | C.
    bn = BayesianNetwork()
    for n in ["A", "B", "C"]:
        bn.add_node(n)
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.set_cpd(TabularCPD("A", (), {(): {"T": 0.5, "F": 0.5}}))
    bn.set_cpd(TabularCPD("B", (), {(): {"T": 0.5, "F": 0.5}}))
    bn.set_cpd(TabularCPD(
        "C", ("A", "B"),
        {("T", "T"): {"T": 0.9, "F": 0.1},
         ("T", "F"): {"T": 0.5, "F": 0.5},
         ("F", "T"): {"T": 0.5, "F": 0.5},
         ("F", "F"): {"T": 0.1, "F": 0.9}},
    ))
    assert bn.is_d_separated(["A"], ["B"], [])
    assert not bn.is_d_separated(["A"], ["B"], ["C"])


def test_d_separation_descendant_also_opens_vstructure():
    # In A -> C <- B with C -> D, observing D also opens the collider.
    bn = BayesianNetwork()
    for n in ["A", "B", "C", "D"]:
        bn.add_node(n)
    bn.add_edge("A", "C"); bn.add_edge("B", "C"); bn.add_edge("C", "D")
    assert not bn.is_d_separated(["A"], ["B"], ["D"])


# ---------------------------------------------------------------------------
# MarkovNetwork
# ---------------------------------------------------------------------------
def test_markov_partition_function():
    mn = MarkovNetwork()
    mn.add_factor(("X",), {("a",): 2.0, ("b",): 1.0})
    mn.add_factor(("Y",), {("a",): 1.0, ("b",): 3.0})
    mn.add_factor(("X", "Y"),
                  {("a", "a"): 0.5,
                   ("a", "b"): 1.0,
                   ("b", "a"): 1.0,
                   ("b", "b"): 0.5})
    domains = {"X": ["a", "b"], "Y": ["a", "b"]}
    Z = mn.partition_function(domains)
    # Direct enumeration:
    manual = 0.0
    for x in ["a", "b"]:
        for y in ["a", "b"]:
            manual += (
                {("a",): 2.0, ("b",): 1.0}[(x,)]
                * {("a",): 1.0, ("b",): 3.0}[(y,)]
                * {("a", "a"): 0.5, ("a", "b"): 1.0,
                   ("b", "a"): 1.0, ("b", "b"): 0.5}[(x, y)]
            )
    assert math.isclose(Z, manual, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Local structures
# ---------------------------------------------------------------------------
def test_deterministic_cpd():
    cpd = DeterministicCPD("Y", ("A", "B"), lambda a, b: a and b)
    assert cpd.prob(True, (True, True)) == 1.0
    assert cpd.prob(False, (True, False)) == 1.0


def test_multiplexer_cpd():
    cpd = MultiplexerCPD("Y", "S", ("A", "B", "C"), domain=[0, 1, 2])
    # S=1 -> B
    assert cpd.prob("x", (1, "x", "x", "x")) == 1.0
    assert cpd.prob("y", (1, "x", "y", "x")) == 1.0
    assert cpd.prob("y", (0, "x", "y", "x")) == 0.0


def test_noisy_or_cpd():
    cpd = NoisyORCPD("Y", ("A", "B"), q=(0.9, 0.8), leak=0.0)
    # P(Y=False) = (1-0.9)(1-0.8) when both parents true
    assert math.isclose(cpd.prob(False, (True, True)), 0.1 * 0.2, rel_tol=1e-12)
    # All parents false -> P(Y=True) = 0 (no leak)
    assert cpd.prob(True, (False, False)) == 0.0
    # One parent true -> P(Y=True) = q_i
    assert math.isclose(cpd.prob(True, (True, False)), 0.9, rel_tol=1e-12)


def test_noisy_max_cpd_distribution_sums_to_one():
    cpd = NoisyMAXCPD(
        "Y", ("A", "B"),
        distributions={
            "A": {0: 0.2, 1: 0.8},
            "B": {0: 0.6, 1: 0.4},
        },
        default=0,
    )
    for pa in [(True, True), (True, False), (False, True), (False, False)]:
        d = cpd.distribution(pa)
        assert math.isclose(sum(d.values()), 1.0, rel_tol=1e-9)


def test_logistic_cpd_matches_sigmoid():
    cpd = LogisticCPD("Y", ("X",), weights=(1.0,), bias=0.0)
    p_true = cpd.prob(True, (1.0,))
    assert math.isclose(p_true, 1.0 / (1.0 + math.exp(-1.0)), rel_tol=1e-12)


def test_linear_gaussian_cpd_mean_and_pdf():
    cpd = LinearGaussianCPD(
        "Y", ("X",), weights=(2.0,), bias=1.0, variance=0.5
    )
    assert cpd.mean((3.0,)) == 7.0
    # pdf integrates to approximately 1 over a fine grid.
    import numpy as np
    xs = np.linspace(-20, 30, 5000)
    dx = xs[1] - xs[0]
    integral = sum(cpd.pdf(float(x), (3.0,)) * dx for x in xs)
    assert math.isclose(integral, 1.0, rel_tol=1e-3, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# Plate model
# ---------------------------------------------------------------------------
def test_plate_grounding():
    plate = PlateModel(index_name="i", index_values=[0, 1, 2])
    plate.add_template(
        "X_{i}", (),
        lambda n, _p: TabularCPD(n, (), {(): {0: 0.5, 1: 0.5}}),
    )
    plate.add_template(
        "Y_{i}", ("X_{i}",),
        lambda n, p: TabularCPD(
            n, p,
            {(0,): {0: 0.9, 1: 0.1}, (1,): {0: 0.1, 1: 0.9}},
        ),
    )
    bn = plate.ground()
    assert sorted(bn.nodes()) == ["X_0", "X_1", "X_2", "Y_0", "Y_1", "Y_2"]
    assert sorted(bn.edges()) == [
        ("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")
    ]


# ---------------------------------------------------------------------------
# Dynamic Bayesian Network
# ---------------------------------------------------------------------------
def test_dbn_unroll_sizes_and_edges():
    dbn = DynamicBayesianNetwork(state_vars=["S"])
    dbn.set_initial(TabularCPD("S", (), {(): {"a": 0.5, "b": 0.5}}))
    dbn.set_transition(
        TabularCPD(
            "S", ("S",),
            {("a",): {"a": 0.7, "b": 0.3}, ("b",): {"a": 0.4, "b": 0.6}},
        )
    )
    dbn.add_observation(
        "O", TabularCPD(
            "O", ("S",),
            {("a",): {"x": 0.9, "y": 0.1}, ("b",): {"x": 0.2, "y": 0.8}},
        )
    )
    bn = dbn.unroll(T=3)
    # Nodes
    assert set(bn.nodes()) == {
        "S_0", "S_1", "S_2", "O_0", "O_1", "O_2"
    }
    # Intra-slice observation edges
    for t in range(3):
        assert (f"S_{t}", f"O_{t}") in bn.edges()
    # Inter-slice transition edges
    assert ("S_0", "S_1") in bn.edges()
    assert ("S_1", "S_2") in bn.edges()
    assert ("S_2", "S_3") not in bn.edges()


def test_dbn_set_transition_accepts_var_and_cpd():
    dbn = DynamicBayesianNetwork(state_vars=["S"])
    dbn.set_initial(TabularCPD("S", (), {(): {"a": 0.5, "b": 0.5}}))
    cpd = TabularCPD(
        "S", ("S",), {("a",): {"a": 0.7, "b": 0.3}, ("b",): {"a": 0.4, "b": 0.6}}
    )
    dbn.set_transition("S", cpd)          # explicit form
    assert dbn.transition_cpds["S"] is cpd
    dbn2 = DynamicBayesianNetwork(state_vars=["S"])
    dbn2.set_transition(cpd)              # inferred from cpd.variable
    assert dbn2.transition_cpds["S"] is cpd


def test_tree_cpd_matches_intermediate_prefix():
    from pgm_handbook.representation import TreeCPD

    cpd = TreeCPD(
        "X",
        ["A", "B", "C"],
        {(1, 1): {"y": 1.0}, (0,): {"z": 1.0}},
        default={"n": 1.0},
    )
    assert cpd.distribution((1, 1, 0)) == {"y": 1.0}  # length-2 prefix
    assert cpd.distribution((0, 1, 1)) == {"z": 1.0}  # length-1 prefix
    assert cpd.distribution((1, 0, 0)) == {"n": 1.0}  # falls back to default
