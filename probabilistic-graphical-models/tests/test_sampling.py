"""Tests for importance sampling, inverse transform, rejection, MCMC."""

from __future__ import annotations

import math

import numpy as np
import pytest

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
# Importance sampling
# ---------------------------------------------------------------------------
def test_importance_sampling_mean_estimates():
    # E[X] under N(0, 1) using a N(1, 2^2) proposal -> 0.
    rng = np.random.default_rng(0)
    target = lambda x: math.exp(-0.5 * x * x)
    q = lambda x: math.exp(-0.5 * ((x - 1.0) / 2.0) ** 2) / 2.0
    sample = lambda r: float(r.normal(1.0, 2.0))

    xs, ws = self_normalized_importance_sampling(
        target, q, sample, 100_000, rng=rng
    )
    est = float(np.mean([w * x for x, w in zip(xs, ws)]))
    assert abs(est) < 0.05


def test_importance_sampling_weights_shapes():
    rng = np.random.default_rng(0)
    sample = lambda r: float(r.normal(0.0, 1.0))
    xs, ws = importance_sampling(
        target_unnorm=lambda x: 1.0,
        proposal_unnorm=lambda x: math.exp(-0.5 * x * x),
        proposal_sample=sample,
        n_samples=200,
        rng=rng,
    )
    assert len(xs) == 200
    assert ws.shape == (200,)


# ---------------------------------------------------------------------------
# Inverse transform
# ---------------------------------------------------------------------------
def test_inverse_transform_exponential():
    rate = 3.0
    rng = np.random.default_rng(0)
    xs = inverse_transform_sample(
        lambda u: -math.log(1.0 - u) / rate, 50_000, rng=rng
    )
    assert abs(xs.mean() - 1.0 / rate) < 0.02
    # Variance of Exponential(rate) is 1 / rate^2.
    assert abs(xs.var() - 1.0 / rate ** 2) < 0.02


def test_inverse_transform_discrete_pmf():
    pmf = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    rng = np.random.default_rng(0)
    xs = inverse_transform_sample_discrete(pmf, 50_000, rng=rng)
    for k, p in pmf.items():
        assert abs(xs.count(k) / len(xs) - p) < 0.01


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------
def test_rejection_sample_box():
    # Sample from N(0, 1) restricted to [-2, 2] using box proposal.
    target = lambda x: (
        math.exp(-0.5 * float(x[0]) ** 2) if -2 <= x[0] <= 2 else 0.0
    )
    rng = np.random.default_rng(0)
    accepted, total = rejection_sample_box(
        target, lower=[-2.0], upper=[2.0], M=1.0,
        n_samples=20_000, rng=rng,
    )
    arr = np.array([a[0] for a in accepted])
    assert abs(arr.mean()) < 0.02
    # Expected variance of N(0,1) truncated symmetrically at ±b:
    #   Var = 1 - 2 b phi(b) / (Phi(b) - Phi(-b))
    b = 2.0
    phi_b = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)
    mass = math.erf(b / math.sqrt(2.0))
    true_var = 1.0 - 2.0 * b * phi_b / mass
    assert abs(arr.var() - true_var) < 0.05
    # Acceptance rate should be in a plausible range.
    assert 0.5 < len(accepted) / total < 1.0


def test_rejection_sample_envelope():
    # Simple 1-D target on [0, 1] proportional to x.
    def target(x: float) -> float:
        return x if 0 <= x <= 1 else 0.0

    def proposal(x: float) -> float:
        return 1.0 if 0 <= x <= 1 else 0.0

    rng = np.random.default_rng(0)
    accepted, total = rejection_sample(
        target, proposal, lambda r: float(r.uniform(0.0, 1.0)),
        M=1.0, n_samples=20_000, rng=rng,
    )
    arr = np.asarray(accepted)
    # Beta(2, 1) mean = 2/3.
    assert abs(arr.mean() - 2.0 / 3.0) < 0.02


# ---------------------------------------------------------------------------
# MCMC — Metropolis
# ---------------------------------------------------------------------------
def test_metropolis_hastings_gaussian():
    def target(x: float) -> float:
        return math.exp(-0.5 * x * x)

    def proposal(x: float, rng):
        return float(x + rng.normal(0.0, 1.0))

    def q(new: float, old: float) -> float:
        return math.exp(-0.5 * (new - old) ** 2)

    rng = np.random.default_rng(0)
    chain = metropolis_hastings(
        target, proposal, q, x0=0.0,
        n_samples=40_000, burn_in=1000, thin=2, rng=rng,
    )
    arr = np.asarray(chain)
    assert abs(arr.mean()) < 0.03
    assert abs(arr.var() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# MCMC — Gibbs
# ---------------------------------------------------------------------------
def test_gibbs_bivariate_gaussian():
    rho = 0.6

    def cond_x(state, rng):
        return float(rng.normal(rho * state["y"], math.sqrt(1 - rho ** 2)))

    def cond_y(state, rng):
        return float(rng.normal(rho * state["x"], math.sqrt(1 - rho ** 2)))

    rng = np.random.default_rng(0)
    chain = gibbs_sampler(
        {"x": cond_x, "y": cond_y},
        x0={"x": 0.0, "y": 0.0},
        n_samples=30_000, burn_in=500, thin=1, rng=rng,
    )
    xs = np.array([s["x"] for s in chain])
    ys = np.array([s["y"] for s in chain])
    assert abs(xs.mean()) < 0.05
    assert abs(ys.mean()) < 0.05
    assert abs(np.corrcoef(xs, ys)[0, 1] - rho) < 0.05


def test_rejection_sample_box_acceptance_rate_matches_theory():
    # Target exp(-x^2/2) on [-1, 1], M = 1 bounds the target on the box:
    # acceptance = Z / (M * volume) = 1.7112 / 2.
    target = lambda x: math.exp(-0.5 * float(x[0]) ** 2)
    rng = np.random.default_rng(1)
    accepted, total = rejection_sample_box(
        target, lower=[-1.0], upper=[1.0], M=1.0, n_samples=20_000, rng=rng
    )
    assert abs(len(accepted) / total - 0.8556) < 0.02


def test_rejection_sample_raises_instead_of_looping_forever():
    with pytest.raises(RuntimeError):
        rejection_sample(
            target_unnorm=lambda x: 0.0,
            proposal_unnorm=lambda x: 1.0,
            proposal_sample=lambda r: r.uniform(),
            M=1.0,
            n_samples=5,
            rng=np.random.default_rng(0),
            max_proposals=100,
        )


def test_metropolis_hastings_irreversible_proposal_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples = metropolis_hastings(
            target_unnorm=lambda x: 1.0,
            proposal_sample=lambda x, r: x + 1,
            proposal_density=lambda a, b: 1.0 if a == b + 1 else 0.0,
            x0=0,
            n_samples=5,
            rng=np.random.default_rng(0),
        )
    assert samples == [0] * 5  # every irreversible move is rejected


def test_metropolis_hastings_docstring_renders():
    assert r"\tilde{p}" in metropolis_hastings.__doc__
