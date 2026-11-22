"""Stochastic Gradient Langevin Dynamics.

Update:
    theta_{t+1} = theta_t + (eps_t / 2) * grad log p(theta_t | D)
                  + N(0, eps_t I)

where ``grad log p(theta | D) = -grad L(theta) - lambda * theta`` (Gaussian
prior with precision ``lambda``).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch.optim import Optimizer


class SGLD(Optimizer):
    """SGLD optimiser/sampler."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        prior_precision: float = 1.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, prior_precision=prior_precision)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            prior_prec = group["prior_precision"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # d(-log posterior)/dtheta = grad_loss + prior_prec * theta
                grad = p.grad + prior_prec * p.data
                noise = torch.randn_like(p) * math.sqrt(lr)
                p.data.add_(grad, alpha=-0.5 * lr).add_(noise)
        return loss


# ---------------------------------------------------------------------------
# Helpers: collect posterior samples and form an ensemble predictive
# ---------------------------------------------------------------------------
def sgld_posterior_samples(
    params: Sequence[torch.nn.Parameter],
    neg_log_lik: Callable[[], torch.Tensor],
    n_steps: int,
    lr: float,
    prior_precision: float = 1.0,
    burn_in: int = 0,
    thin: int = 1,
) -> List[List[torch.Tensor]]:
    """Run SGLD and return thinned post-burn-in snapshots of ``params``.

    ``neg_log_lik()`` must return the *full-data-scaled* negative
    log-likelihood of a (mini)batch, i.e. ``-(N / B) * sum_batch log p(y|x,w)``;
    the Gaussian prior enters through ``prior_precision``.  Parameters are
    left at the final SGLD state.
    """
    if thin < 1:
        raise ValueError("thin must be >= 1")
    params = list(params)
    opt = SGLD(params, lr=lr, prior_precision=prior_precision)
    snapshots: List[List[torch.Tensor]] = []
    for step in range(n_steps):
        opt.zero_grad()
        neg_log_lik().backward()
        opt.step()
        if step >= burn_in and (step - burn_in) % thin == 0:
            snapshots.append([p.detach().clone() for p in params])
    return snapshots


@torch.no_grad()
def ensemble_predict(
    forward: Callable[[torch.Tensor], torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    snapshots: Sequence[Sequence[torch.Tensor]],
    x: torch.Tensor,
    noise_std: float = 0.0,
) -> dict:
    """Predictive mean / epistemic / aleatoric / total std over SGLD samples.

    ``forward(x)`` is evaluated with ``params`` set to each snapshot in turn
    (original values are restored afterwards).
    """
    params = list(params)
    backup = [p.detach().clone() for p in params]
    preds = []
    for snap in snapshots:
        for p, s in zip(params, snap):
            p.copy_(s)
        preds.append(forward(x))
    for p, b in zip(params, backup):
        p.copy_(b)
    stack = torch.stack(preds)
    epi_var = stack.var(0, unbiased=False)
    ale_var = torch.full_like(epi_var, noise_std ** 2)
    return {
        "mean": stack.mean(0),
        "epistemic_std": epi_var.sqrt(),
        "aleatoric_std": ale_var.sqrt(),
        "total_std": (epi_var + ale_var).sqrt(),
    }
