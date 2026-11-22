"""Example 03 — approximate inference: importance, inverse-transform,
rejection, Metropolis-Hastings, and Gibbs.

Each section compares the Monte-Carlo estimate against a closed-form
analytical result to show the convergence behaviour.
"""

from __future__ import annotations

import math

import numpy as np

from pgm_handbook.inference.importance import (
    importance_sampling,
    self_normalized_importance_sampling,
)
from pgm_handbook.inference.inverse_transform import (
    inverse_transform_sample,
    inverse_transform_sample_discrete,
)
from pgm_handbook.inference.rejection import (
    rejection_sample,
    rejection_sample_box,
)
from pgm_handbook.inference.mcmc import metropolis_hastings, gibbs_sampler


# ---------------------------------------------------------------------------
# 1. Importance sampling: E[x^2] under N(0, 1), proposal N(1, 2^2)
# ---------------------------------------------------------------------------
def demo_importance_sampling() -> None:
    print("=" * 70)
    print("1. Importance sampling — E[X^2] under N(0,1)")
    print("=" * 70)

    def target(x: float) -> float:
        return math.exp(-0.5 * x ** 2)  # proportional to N(0,1)

    def proposal(x: float) -> float:
        return math.exp(-0.5 * ((x - 1.0) / 2.0) ** 2) / 2.0

    def sample(rng):
        return float(rng.normal(1.0, 2.0))

    n = 200_000
    xs, ws = importance_sampling(target, proposal, sample, n)
    est_is = float(np.mean([w * (x ** 2) for x, w in zip(xs, ws)])
                   / np.mean(ws))

    xs, ws = self_normalized_importance_sampling(target, proposal, sample, n)
    est_snis = float(np.sum([w * (x ** 2) for x, w in zip(xs, ws)]))  # weights sum to 1

    print(f"  Analytical E[X^2]        = 1.000000")
    print(f"  Self-normalised IS est.  = {est_snis:.4f}")
    print(f"  (both estimators should be close to 1)")
    assert abs(est_snis - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 2. Inverse-transform sampling
# ---------------------------------------------------------------------------
def demo_inverse_transform() -> None:
    print()
    print("=" * 70)
    print("2. Inverse-transform sampling")
    print("=" * 70)

    # Continuous: Exponential(rate=2).  F^{-1}(u) = -ln(1-u)/2
    rate = 2.0
    xs = inverse_transform_sample(lambda u: -math.log(1.0 - u) / rate, 100_000)
    print(f"  Exponential(rate={rate}): sample mean = {xs.mean():.4f}"
          f" (theory 1/{rate})")
    assert abs(xs.mean() - 1.0 / rate) < 0.02

    # Discrete: a 4-sided die with a non-uniform pmf.
    pmf = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    ds = inverse_transform_sample_discrete(pmf, 100_000)
    empirical = {k: ds.count(k) / len(ds) for k in pmf}
    print("  Die pmf empirical:", {k: round(v, 3) for k, v in empirical.items()})
    for k, p in pmf.items():
        assert abs(empirical[k] - p) < 0.01


# ---------------------------------------------------------------------------
# 3. Rejection sampling: sample from a truncated Gaussian on [-1, 1]
# ---------------------------------------------------------------------------
def demo_rejection() -> None:
    print()
    print("=" * 70)
    print("3. Rejection sampling — standard normal truncated to [-1, 1]")
    print("=" * 70)

    # Unnormalised target: N(0,1) restricted to [-1, 1] (indicator)
    def target(x: float) -> float:
        if not (-1.0 <= x <= 1.0):
            return 0.0
        return math.exp(-0.5 * x ** 2)

    def proposal(x: float) -> float:
        return 1.0 / 2.0  # uniform on [-1, 1]

    def sample(rng):
        return float(rng.uniform(-1.0, 1.0))

    # Envelope condition: target(x) <= M * proposal(x).  The unnormalised
    # target peaks at 1 (x = 0) and the proposal density is 0.5, so M = 2.
    accepted, total = rejection_sample(
        target, proposal, sample, M=2.0, n_samples=50_000
    )
    acc = np.asarray(accepted)
    print(f"  Acceptance rate ≈ {len(accepted)/total:.3f}")
    print(f"  Sample mean     = {acc.mean():.4f}  (theory 0)")
    # Var of N(0,1) truncated to [-1, 1]:  1 - 2 phi(1) / (2 Phi(1) - 1)
    phi_1 = math.exp(-0.5) / math.sqrt(2.0 * math.pi)
    mass = math.erf(1.0 / math.sqrt(2.0))  # = 2 Phi(1) - 1
    theory_var = 1 - 2 * phi_1 / mass
    print(f"  Sample var      = {acc.var():.4f}  (theory ≈ {theory_var:.4f})")
    assert abs(acc.mean()) < 0.02
    assert abs(acc.var() - theory_var) < 0.01
    # Expected acceptance = Z / (M * ∫q) = 1.711 / 2 ≈ 0.856
    assert abs(len(accepted) / total - 0.856) < 0.02

    # Bounding-box variant: M bounds the unnormalised target on the box (= 1).
    accepted, total = rejection_sample_box(
        lambda x: target(float(x[0])),
        lower=[-1.0], upper=[1.0], M=1.0, n_samples=20_000,
    )
    box = np.array([a[0] for a in accepted])
    print(f"  Box-rejection: {len(accepted)} / {total} accepted"
          f"  (var = {box.var():.4f})")
    assert abs(box.var() - theory_var) < 0.02


# ---------------------------------------------------------------------------
# 4. MCMC: Metropolis-Hastings on a 1-D mixture of Gaussians
# ---------------------------------------------------------------------------
def demo_mh() -> None:
    print()
    print("=" * 70)
    print("4. Metropolis-Hastings — bimodal 0.5·N(-2, 0.5²) + 0.5·N(2, 0.5²)")
    print("=" * 70)

    def target(x: float) -> float:
        return 0.5 * math.exp(-0.5 * ((x + 2) / 0.5) ** 2) + \
               0.5 * math.exp(-0.5 * ((x - 2) / 0.5) ** 2)

    def proposal(x: float, rng):
        return float(x + rng.normal(0.0, 0.5))

    def q_density(new: float, old: float) -> float:
        # Gaussian proposal is symmetric: same in both directions.
        return math.exp(-0.5 * ((new - old) / 0.5) ** 2)

    rng = np.random.default_rng(123)
    chain = metropolis_hastings(
        target, proposal, q_density, x0=0.0,
        n_samples=50_000, burn_in=1_000, thin=2, rng=rng,
    )
    chain = np.asarray(chain)
    print(f"  Chain mean   = {chain.mean():.4f}  (theory 0)")
    print(f"  P(x>0)       = {(chain > 0).mean():.4f}  (theory 0.5)")
    # Standard error of the mean: σ/sqrt(n) ≈ sqrt(4.25)/sqrt(50000) ≈ 0.009
    assert abs(chain.mean()) < 0.05
    assert abs((chain > 0).mean() - 0.5) < 0.02


# ---------------------------------------------------------------------------
# 5. Gibbs on a bivariate Gaussian
# ---------------------------------------------------------------------------
def demo_gibbs() -> None:
    print()
    print("=" * 70)
    print("5. Gibbs — bivariate Gaussian with correlation ρ=0.7")
    print("=" * 70)

    rho = 0.7

    def cond_x(state, rng):
        # x | y ~ N(rho*y, 1 - rho²)
        return float(rng.normal(rho * state["y"], math.sqrt(1 - rho ** 2)))

    def cond_y(state, rng):
        return float(rng.normal(rho * state["x"], math.sqrt(1 - rho ** 2)))

    rng = np.random.default_rng(0)
    chain = gibbs_sampler(
        {"x": cond_x, "y": cond_y},
        x0={"x": 0.0, "y": 0.0},
        n_samples=30_000, burn_in=500, thin=2, rng=rng,
    )
    xs = np.array([s["x"] for s in chain])
    ys = np.array([s["y"] for s in chain])
    empirical_rho = float(np.corrcoef(xs, ys)[0, 1])
    print(f"  Sample means: <x>={xs.mean():.3f}, <y>={ys.mean():.3f}")
    print(f"  Sample ρ      = {empirical_rho:.3f}  (theory {rho})")
    assert abs(xs.mean()) < 0.05
    assert abs(ys.mean()) < 0.05
    assert abs(empirical_rho - rho) < 0.05


def main() -> None:
    demo_importance_sampling()
    demo_inverse_transform()
    demo_rejection()
    demo_mh()
    demo_gibbs()
    print("\nAll approximate-inference demos completed successfully.")


if __name__ == "__main__":
    main()
