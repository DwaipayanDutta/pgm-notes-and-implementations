"""Markov chain Monte Carlo: Metropolis and Metropolis–Hastings."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping

import numpy as np


def metropolis_hastings(
    target_unnorm: Callable,
    proposal_sample: Callable,
    proposal_density: Callable,
    x0,
    n_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.Generator | None = None,
) -> List:
    r"""Metropolis–Hastings sampler.

    Parameters
    ----------
    target_unnorm : callable
        Unnormalised target density ``\tilde{p}(x)``.
    proposal_sample : callable
        ``proposal_sample(x, rng)`` returns a proposal given current state.
    proposal_density : callable
        ``proposal_density(x_new, x_old)`` returns ``q(x_new | x_old)``.
    """
    rng = rng or np.random.default_rng()
    x = x0
    current_log = float(np.log(max(target_unnorm(x), 1e-300)))

    samples: List = []
    total = burn_in + n_samples * thin
    for i in range(total):
        x_prop = proposal_sample(x, rng)
        prop_log = float(np.log(max(target_unnorm(x_prop), 1e-300)))
        q_fwd = proposal_density(x_prop, x)
        q_bwd = proposal_density(x, x_prop)
        if q_fwd <= 0.0 or q_bwd <= 0.0:
            # q_bwd = 0: the move cannot be reversed, so detailed balance
            # requires rejecting it.
            accept_prob = 0.0
        else:
            log_ratio = prop_log - current_log + np.log(q_bwd) - np.log(q_fwd)
            accept_prob = min(1.0, float(np.exp(log_ratio)))
        if rng.uniform() < accept_prob:
            x = x_prop
            current_log = prop_log
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(x)
    return samples


def gibbs_sampler(
    full_conditionals: Mapping[str, Callable],
    x0: Mapping,
    n_samples: int,
    burn_in: int = 0,
    thin: int = 1,
    rng: np.random.Generator | None = None,
) -> List[Dict]:
    """Coordinate-wise Gibbs sampler.

    ``full_conditionals[var](state, rng)`` returns the new value for ``var``.
    """
    rng = rng or np.random.default_rng()
    state = dict(x0)
    samples: List[Dict] = []
    total = burn_in + n_samples * thin
    for i in range(total):
        for var, fc in full_conditionals.items():
            state[var] = fc(state, rng)
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(dict(state))
    return samples
