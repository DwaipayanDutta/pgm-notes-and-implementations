"""Example 02 — compact local CPDs, plate models and dynamic Bayesian networks.

Demonstrates:
  * DeterministicCPD / MultiplexerCPD / NoisyORCPD / NoisyMAXCPD
    / LogisticCPD / LinearGaussianCPD
  * PlateModel grounding a replicated template into a BayesianNetwork
  * DynamicBayesianNetwork unrolling an HMM-style 2-slice model
"""

from __future__ import annotations

import math

import numpy as np

from pgm_handbook.representation.bayesian_network import (
    BayesianNetwork,
    TabularCPD,
)
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
from pgm_handbook.inference.exact import enumeration_ask


# ---------------------------------------------------------------------------
# 1. Local CPDs
# ---------------------------------------------------------------------------
def demo_local_cpds() -> None:
    print("=" * 70)
    print("1. Local CPDs")
    print("=" * 70)

    # Deterministic
    det = DeterministicCPD(
        variable="Out", parents=("A", "B"), func=lambda a, b: a and b
    )
    print("Deterministic (A AND B):",
          {v: det.prob(v, (True, True)) for v in [True, False]})

    # Multiplexer
    mux = MultiplexerCPD(
        variable="Out",
        selector="Sel",
        choices=("A", "B", "C"),
        domain=["x", "y", "z"],
    )
    # Sel = 1 -> choose B
    print("Multiplexer (Sel=1, B='y'):",
          {v: mux.prob(v, (1, "x", "y", "z")) for v in ["x", "y", "z"]})

    # Noisy-OR
    nor = NoisyORCPD(
        variable="Fever",
        parents=("Flu", "Cold"),
        q=(0.95, 0.4),
        leak=0.01,
    )
    for fa, ca in [(False, False), (True, False), (False, True), (True, True)]:
        print(f"  NoisyOR Flu={fa} Cold={ca}: P(Fever=True)={nor.prob(True, (fa, ca)):.4f}")

    # Noisy-MAX
    nmax = NoisyMAXCPD(
        variable="Symptom",
        parents=("Flu", "Cold"),
        distributions={
            "Flu":  {0: 0.1, 1: 0.2, 2: 0.7},
            "Cold": {0: 0.6, 1: 0.3, 2: 0.1},
        },
        default=0,
    )
    print("NoisyMAX P(Symptom=1 | Flu=1, Cold=0):",
          f"{nmax.prob(1, (True, False)):.4f}")

    # Logistic
    logit = LogisticCPD(
        variable="Click",
        parents=("Age", "Income"),
        weights=(0.02, 0.0001),
        bias=-1.0,
    )
    print("Logistic P(Click=True | Age=35, Income=50000):",
          f"{logit.prob(True, (35, 50000)):.4f}")

    # Linear-Gaussian
    lg = LinearGaussianCPD(
        variable="Sensor",
        parents=("Temp",),
        weights=(1.0,),
        bias=0.0,
        variance=0.25,
    )
    rng = np.random.default_rng(0)
    samples = [lg.sample((20.0,), rng) for _ in range(5)]
    print("LinearGaussian mean(20) =", lg.mean((20.0,)),
          "| sample:", [round(s, 3) for s in samples])


# ---------------------------------------------------------------------------
# 2. Plate model: student grades
# ---------------------------------------------------------------------------
def demo_plate() -> None:
    print()
    print("=" * 70)
    print("2. Plate model — 3 students with Difficulty/Intelligence -> Grade")
    print("=" * 70)

    plate = PlateModel(index_name="i", index_values=[0, 1, 2])

    # Prior on Intelligence
    plate.add_template(
        "Int_{i}",
        (),
        lambda name, _parents: TabularCPD(
            name, (), {(): {0: 0.7, 1: 0.3}}
        ),
    )
    # Prior on Difficulty
    plate.add_template(
        "Diff_{i}",
        (),
        lambda name, _parents: TabularCPD(
            name, (), {(): {0: 0.4, 1: 0.6}}
        ),
    )
    # Grade | Intelligence, Difficulty
    grade_table = {
        (0, 0): {0: 0.3, 1: 0.4, 2: 0.3},
        (0, 1): {0: 0.05, 1: 0.25, 2: 0.7},
        (1, 0): {0: 0.9, 1: 0.08, 2: 0.02},
        (1, 1): {0: 0.5, 1: 0.3, 2: 0.2},
    }
    plate.add_template(
        "Grade_{i}",
        ("Int_{i}", "Diff_{i}"),
        lambda name, parents: TabularCPD(name, parents, grade_table),
    )

    bn = plate.ground()
    print("Grounded nodes:", sorted(bn.nodes()))
    print("Grounded edges:", sorted(bn.edges()))
    assert len(bn.nodes()) == 9
    assert len(bn.edges()) == 6

    # Every student's marginal P(Grade) should match the prior-averaged table.
    # (First student only — conditional independence on the plate.)
    p_grade = enumeration_ask("Grade_0", {}, bn)
    print("P(Grade_0):", {k: round(v, 3) for k, v in p_grade.items()})


# ---------------------------------------------------------------------------
# 3. HMM as a dynamic Bayesian network
# ---------------------------------------------------------------------------
def demo_dbn() -> None:
    print()
    print("=" * 70)
    print("3. Dynamic Bayesian Network — 3-step HMM")
    print("=" * 70)

    dbn = DynamicBayesianNetwork(state_vars=["Weather"])
    # Initial prior
    dbn.set_initial(
        TabularCPD(
            "Weather",
            (),
            {(): {"Sun": 0.6, "Rain": 0.4}},
        )
    )
    # Transition (parents are in slice t-1, remapped automatically)
    dbn.set_transition(
        "Weather",
        TabularCPD(
            "Weather",
            ("Weather",),
            {
                ("Sun",):  {"Sun": 0.7, "Rain": 0.3},
                ("Rain",): {"Sun": 0.4, "Rain": 0.6},
            },
        ),
    )
    # Observation (same slice)
    dbn.add_observation(
        "Umbrella",
        TabularCPD(
            "Umbrella",
            ("Weather",),
            {
                ("Sun",):  {"yes": 0.1, "no": 0.9},
                ("Rain",): {"yes": 0.8, "no": 0.2},
            },
        ),
    )

    bn = dbn.unroll(T=3)
    print("Unrolled nodes:", sorted(bn.nodes()))
    print("Unrolled edges:", sorted(bn.edges()))
    assert "Weather_0" in bn.nodes()
    assert "Weather_2" in bn.nodes()
    assert ("Weather_0", "Weather_1") in bn.edges()
    assert ("Weather_1", "Weather_2") in bn.edges()

    # Filtered P(Weather_1 | Umbrella_1 = yes)
    p = enumeration_ask("Weather_1", {"Umbrella_1": "yes"}, bn)
    print("P(Weather_1 | Umbrella_1=yes):",
          {k: round(v, 3) for k, v in p.items()})

    # Smoothing:  P(Weather_1 | Umbrella_0=yes, Umbrella_1=yes)
    p2 = enumeration_ask(
        "Weather_1", {"Umbrella_0": "yes", "Umbrella_1": "yes"}, bn
    )
    print("P(Weather_1 | Umbrella_0=yes, Umbrella_1=yes):",
          {k: round(v, 3) for k, v in p2.items()})

    # Sanity: seeing an umbrella today increases P(Rain).
    assert p["Rain"] > 0.4
    assert p2["Rain"] > 0.4
    print("✓ Umbrella evidence raises P(Rain)")


def main() -> None:
    demo_local_cpds()
    demo_plate()
    demo_dbn()


if __name__ == "__main__":
    main()
