"""Rejection sampling (envelope and bounding-box variants)."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np


def rejection_sample(
    target_unnorm: Callable,
    proposal_unnorm: Callable,
    proposal_sample: Callable,
    M: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
    max_proposals: int | None = None,
) -> Tuple[List, int]:
    """Rejection sampling with envelope ``target(x) <= M * proposal_unnorm(x)``.

    Returns ``(accepted_samples, total_proposals)``.

    ``max_proposals`` caps the number of draws (default ``1000 * n_samples``,
    minimum 10,000) so that a bad envelope or a target that is zero
    everywhere raises ``RuntimeError`` instead of looping forever.
    """
    rng = rng or np.random.default_rng()
    if max_proposals is None:
        max_proposals = max(10_000, 1000 * n_samples)
    accepted: List = []
    total = 0
    while len(accepted) < n_samples:
        if total >= max_proposals:
            raise RuntimeError(
                f"rejection_sample accepted only {len(accepted)}/{n_samples} "
                f"samples in {total} proposals; check M and the envelope"
            )
        x = proposal_sample(rng)
        total += 1
        u = rng.uniform(0.0, M * proposal_unnorm(x))
        if u <= target_unnorm(x):
            accepted.append(x)
    return accepted, total


def rejection_sample_box(
    target_unnorm: Callable,
    lower: Sequence[float],
    upper: Sequence[float],
    M: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
    max_proposals: int | None = None,
) -> Tuple[List, int]:
    """Bounding-box rejection sampling (proposal = uniform on the box).

    ``M`` must be an upper bound of ``target_unnorm`` over the box, i.e. the
    envelope is the constant function ``M``.  The acceptance rate is then
    ``Z / (M * volume)``.
    """
    rng = rng or np.random.default_rng()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    def sample(_rng):
        return _rng.uniform(lower, upper)

    return rejection_sample(
        target_unnorm=target_unnorm,
        proposal_unnorm=lambda _x: 1.0,
        proposal_sample=sample,
        M=M,
        n_samples=n_samples,
        rng=rng,
        max_proposals=max_proposals,
    )
