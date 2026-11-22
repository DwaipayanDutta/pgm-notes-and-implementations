"""Smirnov / inverse-transform sampling."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np


def inverse_transform_sample(
    inverse_cdf: Callable[[float], float],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Continuous inverse-transform sampling: ``X = F^{-1}(U)``."""
    rng = rng or np.random.default_rng()
    u = rng.uniform(0.0, 1.0, n_samples)
    return np.array([inverse_cdf(float(ui)) for ui in u])


def inverse_transform_sample_discrete(
    pmf: Dict,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> List:
    """Discrete inverse-transform sampling via interval lookup."""
    rng = rng or np.random.default_rng()
    values = list(pmf.keys())
    probs = np.array([pmf[v] for v in values], dtype=float)
    probs = probs / probs.sum()
    cdf = np.cumsum(probs)
    u = rng.uniform(0.0, 1.0, n_samples)
    idx = np.searchsorted(cdf, u, side="right")
    idx = np.clip(idx, 0, len(values) - 1)
    return [values[i] for i in idx]
