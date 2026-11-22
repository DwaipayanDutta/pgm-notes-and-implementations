"""Importance sampling and self-normalised importance sampling."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def importance_sampling(
    target_unnorm: Callable,
    proposal_unnorm: Callable,
    proposal_sample: Callable,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> Tuple[List, np.ndarray]:
    """Standard importance sampling.

    Returns ``(samples, weights)``.  To estimate ``E_p[f]`` use
    ``sum(w_i f(x_i)) / sum(w_i)`` (self-normalised form).
    """
    rng = rng or np.random.default_rng()
    samples: List = []
    weights = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        x = proposal_sample(rng)
        samples.append(x)
        q = proposal_unnorm(x)
        weights[i] = target_unnorm(x) / q if q > 0 else 0.0
    return samples, weights


def self_normalized_importance_sampling(
    target_unnorm: Callable,
    proposal_unnorm: Callable,
    proposal_sample: Callable,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> Tuple[List, np.ndarray]:
    """Self-normalised IS.  Weights sum to 1."""
    samples, weights = importance_sampling(
        target_unnorm, proposal_unnorm, proposal_sample, n_samples, rng
    )
    Z = weights.sum()
    if Z <= 0:
        raise ValueError("All weights are zero; proposal is unsuitable")
    return samples, weights / Z
