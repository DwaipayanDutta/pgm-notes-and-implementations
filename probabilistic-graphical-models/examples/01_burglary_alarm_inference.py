"""Example 01 — classic Burglary / Earthquake / Alarm / John / Mary network.

Reproduces the textbook d-separation and inference result:

    P(B = True | J = True, M = True) ≈ 0.284

and verifies enumeration and variable elimination agree exactly.
"""

from __future__ import annotations

from pgm_handbook.representation.bayesian_network import (
    BayesianNetwork,
    TabularCPD,
)
from pgm_handbook.inference.exact import (
    enumeration_ask,
    variable_elimination,
)


def build_burglary_network() -> BayesianNetwork:
    """Build the classic Pearl (1988) burglary network."""
    bn = BayesianNetwork()

    for node in ("Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"):
        bn.add_node(node)

    bn.set_cpd(
        TabularCPD(
            "Burglary",
            (),
            {(): {True: 0.001, False: 0.999}},
        )
    )
    bn.set_cpd(
        TabularCPD(
            "Earthquake",
            (),
            {(): {True: 0.002, False: 0.998}},
        )
    )
    bn.set_cpd(
        TabularCPD(
            "Alarm",
            ("Burglary", "Earthquake"),
            {
                (True, True):  {True: 0.95, False: 0.05},
                (True, False): {True: 0.94, False: 0.06},
                (False, True): {True: 0.29, False: 0.71},
                (False, False): {True: 0.001, False: 0.999},
            },
        )
    )
    bn.set_cpd(
        TabularCPD(
            "JohnCalls",
            ("Alarm",),
            {
                (True,):  {True: 0.90, False: 0.10},
                (False,): {True: 0.05, False: 0.95},
            },
        )
    )
    bn.set_cpd(
        TabularCPD(
            "MaryCalls",
            ("Alarm",),
            {
                (True,):  {True: 0.70, False: 0.30},
                (False,): {True: 0.01, False: 0.99},
            },
        )
    )
    return bn


def main() -> None:
    bn = build_burglary_network()
    print("BN structure:", bn.nodes())
    print("BN edges   :", bn.edges())

    # ---------- 1. Posterior via enumeration ----------
    evidence = {"JohnCalls": True, "MaryCalls": True}
    post_enum = enumeration_ask("Burglary", evidence, bn)
    print("\nEnumeration  P(B | J, M):", post_enum)

    # ---------- 2. Posterior via variable elimination ----------
    # Order chosen to demonstrate that factors get combined sensibly.
    order = ["JohnCalls", "MaryCalls", "Alarm", "Earthquake"]
    post_ve = variable_elimination("Burglary", evidence, bn, order)
    print("Var. Elim.    P(B | J, M):", post_ve)

    # ---------- 3. Verification ----------
    p_true = post_enum[True]
    print(f"\nP(B=True | J=True, M=True) = {p_true:.6f}")
    assert abs(p_true - 0.284) < 1e-3, f"Expected ~0.284, got {p_true}"
    assert abs(post_enum[True] - post_ve[True]) < 1e-9
    assert abs(post_enum[False] - post_ve[False]) < 1e-9
    print("✓ Enumeration matches Variable Elimination")
    print("✓ Posterior ≈ 0.284 (textbook value)")

    # ---------- 4. Evidence changes the picture ----------
    # Knowing John called but *not* Mary makes burglary more likely than
    # the trivial prior — but less so than when both call.
    p_j_only = enumeration_ask("Burglary", {"JohnCalls": True}, bn)
    p_prior = enumeration_ask("Burglary", {}, bn)
    print(f"\nP(B=True | J=True) = {p_j_only[True]:.6f}")
    print(f"P(B=True)          = {p_prior[True]:.6f}")
    assert p_j_only[True] > p_prior[True]
    assert p_j_only[True] < p_true
    print("✓ Single-call posterior lies between prior and joint-call posterior")

    # ---------- 5. d-separation sanity checks ----------
    # John and Mary are d-separated given Alarm.
    assert bn.is_d_separated(["JohnCalls"], ["MaryCalls"], ["Alarm"])
    # ...but *dependent* when Alarm is unobserved (they share a parent).
    assert not bn.is_d_separated(["JohnCalls"], ["MaryCalls"], [])
    # Alarm *explains away*: given Alarm, evidence of Earthquake reduces
    # belief in Burglary.
    p_b_alarm = enumeration_ask("Burglary", {"Alarm": True}, bn)
    p_b_alarm_eq = enumeration_ask(
        "Burglary", {"Alarm": True, "Earthquake": True}, bn
    )
    print(f"\nP(B=True | Alarm=True)                      = {p_b_alarm[True]:.6f}")
    print(f"P(B=True | Alarm=True, Earthquake=True)     = {p_b_alarm_eq[True]:.6f}")
    assert p_b_alarm_eq[True] < p_b_alarm[True]
    print("✓ Explaining-away confirmed (v-structure)")


if __name__ == "__main__":
    main()
