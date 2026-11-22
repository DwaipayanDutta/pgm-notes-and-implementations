"""Bayesian neural network layers with Gaussian priors.

The posterior over weights is approximated by a diagonal Gaussian
``q(w) = N(mu, sigma^2)`` using the reparameterisation trick.  Training
minimises the (variational) free energy ``-E_q[log p(D|w)] + KL(q || p)``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class GaussianPrior:
    """Isotropic Gaussian prior ``N(mean, std^2)``."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def log_prob(self, w: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((w - self.mean) / self.std) ** 2 - math.log(
            self.std * math.sqrt(2.0 * math.pi)
        )


def _kl_diag_gaussian(
    mu: torch.Tensor,
    var: torch.Tensor,
    prior_mean: float,
    prior_var: float,
) -> torch.Tensor:
    """``KL(N(mu, var) || N(prior_mean, prior_var))`` summed over entries."""
    return 0.5 * (
        math.log(prior_var)
        - torch.log(var)
        + (var + (mu - prior_mean) ** 2) / prior_var
        - 1.0
    ).sum()


class BayesianLinear(nn.Module):
    """Fully connected layer whose weights follow a diagonal Gaussian q."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_prior: GaussianPrior | None = None,
        bias_prior: GaussianPrior | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prior = weight_prior or GaussianPrior(0.0, 1.0)
        self.bias_prior = bias_prior or GaussianPrior(0.0, 1.0)

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(
            torch.full((out_features, in_features), -3.0)
        )
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
        # Initialise weights with Kaiming-style scaling to keep activations sane.
        with torch.no_grad():
            fan_in = in_features
            std = 1.0 / math.sqrt(fan_in)
            self.weight_mu.uniform_(-std, std)

    # ------------------------------------------------------------------
    def sample_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_eps = torch.randn_like(self.weight_mu)
        b_eps = torch.randn_like(self.bias_mu)
        w = self.weight_mu + torch.exp(self.weight_log_sigma) * w_eps
        b = self.bias_mu + torch.exp(self.bias_log_sigma) * b_eps
        return w, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self.sample_weights()
        return x @ w.t() + b

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight_mu.t() + self.bias_mu

    def kl_divergence(self) -> torch.Tensor:
        w_var = torch.exp(2.0 * self.weight_log_sigma)
        b_var = torch.exp(2.0 * self.bias_log_sigma)
        kl_w = _kl_diag_gaussian(
            self.weight_mu, w_var, self.weight_prior.mean, self.weight_prior.std ** 2
        )
        kl_b = _kl_diag_gaussian(
            self.bias_mu, b_var, self.bias_prior.mean, self.bias_prior.std ** 2
        )
        return kl_w + kl_b


class BayesianMLP(nn.Module):
    """Multi-layer perceptron with Bayesian weights and output noise.

    ``forward`` returns the sampled output.  ``predictive`` returns an
    MC estimate of the predictive distribution:  ``mean``, ``epistemic_std``,
    ``aleatoric_std`` and ``total_std``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        out_features: int,
        prior: GaussianPrior | None = None,
        initial_log_noise: float = -2.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(BayesianLinear(prev, h, prior, prior))
            layers.append(nn.Tanh())
            prev = h
        layers.append(BayesianLinear(prev, out_features, prior, prior))
        self.net = nn.Sequential(*layers)
        self.log_noise = nn.Parameter(torch.tensor(initial_log_noise))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic pass using the posterior-mean weights."""
        for layer in self.net:
            x = layer.forward_mean(x) if isinstance(layer, BayesianLinear) else layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        total = torch.zeros((), device=self.log_noise.device)
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                total = total + m.kl_divergence()
        return total

    @torch.no_grad()
    def predictive(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> dict[str, torch.Tensor]:
        """Monte-Carlo predictive posterior.

        Returns
        -------
        dict with keys ``mean``, ``epistemic_std``, ``aleatoric_std``,
        ``total_std``.
        """
        was_training = self.training
        self.eval()
        try:
            preds = torch.stack([self.forward(x) for _ in range(n_samples)])
        finally:
            self.train(was_training)
        mean = preds.mean(0)
        epistemic_var = preds.var(0, unbiased=False)
        noise_var = torch.exp(2.0 * self.log_noise)
        aleatoric_var = torch.full_like(epistemic_var, float(noise_var))
        total_var = epistemic_var + aleatoric_var
        return {
            "mean": mean,
            "epistemic_std": epistemic_var.sqrt(),
            "aleatoric_std": aleatoric_var.sqrt(),
            "total_std": total_var.sqrt(),
        }
