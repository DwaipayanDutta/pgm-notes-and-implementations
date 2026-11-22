"""Tests for exact inference: enumeration and variable elimination."""

from __future__ import annotations

import math

import pytest

from pgm_handbook.representation.bayesian_network import (
    BayesianNetwork,
    TabularCPD,
)
from pgm_handbook.inference.exact import (
    enumeration_ask,
    variable_elimination,
)


def _burglary_bn() -> BayesianNetwork:
    bn = BayesianNetwork()
    for node in ("B", "E", "A", "J", "M"):
        bn.add_node(node)
    bn.set_cpd(TabularCPD("B", (), {(): {True: 0.001, False: 0.999}}))
    bn.set_cpd(TabularCPD("E", (), {(): {True: 0.002, False: 0.998}}))
    bn.set_cpd(TabularCPD(
        "A", ("B", "E"),
        {
            (True, True): {True: 0.95, False: 0.05},
            (True, False): {True: 0.94, False: 0.06},
            (False, True): {True: 0.29, False: 0.71},
            (False, False): {True: 0.001, False: 0.999},
        },
    ))
    bn.set_cpd(TabularCPD(
        "J", ("A",),
        {(True,): {True: 0.9, False: 0.1},
         (False,): {True: 0.05, False: 0.95}},
    ))
    bn.set_cpd(TabularCPD(
        "M", ("A",),
        {(True,): {True: 0.7, False: 0.3},
         (False,): {True: 0.01, False: 0.99}},
    ))
    return bn


def test_burglary_posterior_approx_0_284():
    bn = _burglary_bn()
    post = enumeration_ask("B", {"J": True, "M": True}, bn)
    assert abs(post[True] - 0.284) < 1e-3
    assert abs(post[True] + post[False] - 1.0) < 1e-12


def test_enumeration_and_ve_agree():
    bn = _burglary_bn()
    evidence = {"J": True, "M": True}
    post_enum = enumeration_ask("B", evidence, bn)
    for order in (
        ["J", "M", "A", "E"],
        ["E", "A", "J", "M"],
        ["A", "E", "J", "M"],
    ):
        post_ve = variable_elimination("B", evidence, bn, order)
        for v in post_enum:
            assert math.isclose(post_enum[v], post_ve[v], rel_tol=1e-9)


def test_ve_custom_order_incomplete_fills_remaining():
    # An incomplete elimination_order should still eliminate all hidden vars.
    bn = _burglary_bn()
    post = variable_elimination("B", {"J": True}, bn, elimination_order=["M"])
    ref = enumeration_ask("B", {"J": True}, bn)
    for v in ref:
        assert math.isclose(post[v], ref[v], rel_tol=1e-9)


def test_evidence_agreement():
    # P(query | evidence) ∝ P(query, evidence).  Check that joint over query
    # and evidence normalises correctly.
    bn = _burglary_bn()
    post = enumeration_ask("E", {"A": True}, bn)
    joint = {True: 0.0, False: 0.0}
    for e in [True, False]:
        for b in [True, False]:
            for j in [True, False]:
                for m in [True, False]:
                    joint[e] += bn.joint_probability(
                        {"B": b, "E": e, "A": True, "J": j, "M": m}
                    )
    z = sum(joint.values())
    assert z > 0.0
    assert abs(post[True] + post[False] - 1.0) < 1e-12
    assert math.isclose(post[True], joint[True] / z, rel_tol=1e-9)


def test_zero_probability_evidence_raises():
    bn = BayesianNetwork()
    bn.add_node("X")
    bn.set_cpd(TabularCPD("X", (), {(): {"a": 1.0, "b": 0.0}}))
    with pytest.raises(ValueError):
        enumeration_ask("X", {"X": "b"}, bn)


def test_chain_network_ve_vs_enumeration():
    # A -> B -> C -> D.  Simple factor-graph exercise for VE.
    bn = BayesianNetwork()
    for n in "ABCD":
        bn.add_node(n)
    bn.set_cpd(TabularCPD("A", (), {(): {0: 0.4, 1: 0.6}}))
    for parent, child in [("A", "B"), ("B", "C"), ("C", "D")]:
        bn.set_cpd(TabularCPD(
            child, (parent,),
            {(0,): {0: 0.9, 1: 0.1}, (1,): {0: 0.2, 1: 0.8}},
        ))
    post_enum = enumeration_ask("D", {"A": 1}, bn)
    post_ve = variable_elimination("D", {"A": 1}, bn, ["B", "C", "A"])
    for v in post_enum:
        assert math.isclose(post_enum[v], post_ve[v], rel_tol=1e-9)


def test_query_variable_in_evidence_is_point_mass():
    bn = _burglary_bn()
    post = enumeration_ask("A", {"A": True, "J": True}, bn)
    assert post == {True: 1.0, False: 0.0}
    post_ve = variable_elimination("A", {"A": True, "J": True}, bn)
    assert math.isclose(post_ve[True], 1.0) and math.isclose(post_ve[False], 0.0, abs_tol=1e-12)
