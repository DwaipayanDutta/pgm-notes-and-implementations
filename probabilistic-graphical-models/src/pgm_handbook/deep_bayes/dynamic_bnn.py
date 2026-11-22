"""Dynamic / recurrent Bayesian neural network."""

from __future__ import annotations

import torch
import torch.nn as nn

from .bnn import BayesianLinear, GaussianPrior


class DynamicBNN(nn.Module):
    """Recurrent BNN over a sequence with Bayesian weights.

    Latent transition:  ``h_t = tanh(W_trans [x_t; h_{t-1}] + b_trans)``
    Emission:           ``y_t = W_out h_t + b_out``

    All weights follow diagonal Gaussian posteriors; the model supports both
    SGLD sampling and MC predictive inference.  ``forward`` returns one
    sample; ``predictive`` returns MC mean / epistemic / aleatoric std.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        prior: GaussianPrior | None = None,
    ) -> None:
        super().__init__()
        prior = prior or GaussianPrior(0.0, 1.0)
        self.hidden_size = hidden_size
        self.transition = BayesianLinear(
            hidden_size + in_features, hidden_size, prior, prior
        )
        self.emission = BayesianLinear(hidden_size, out_features, prior, prior)
        self.log_noise = nn.Parameter(torch.tensor(-2.0))

    # ------------------------------------------------------------------
    def forward(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one sampled-weight pass through the sequence.

        Parameters
        ----------
        x_seq : torch.Tensor
            Shape ``(batch, seq_len, in_features)``.
        h0 : torch.Tensor, optional
            Initial hidden state, shape ``(batch, hidden_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, out_features)``.
        """
        B, T, _ = x_seq.shape
        h = (
            h0
            if h0 is not None
            else x_seq.new_zeros(B, self.hidden_size)
        )
        outputs = []
        for t in range(T):
            inp = torch.cat([x_seq[:, t], h], dim=-1)
            h = torch.tanh(self.transition(inp))
            outputs.append(self.emission(h))
        return torch.stack(outputs, dim=1)

    def kl_divergence(self) -> torch.Tensor:
        total = torch.zeros((), device=self.log_noise.device)
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                total = total + m.kl_divergence()
        return total

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predictive(
        self,
        x_seq: torch.Tensor,
        n_samples: int = 50,
        h0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Monte-Carlo predictive posterior over the sequence."""
        was_training = self.training
        self.eval()
        try:
            preds = torch.stack(
                [self.forward(x_seq, h0) for _ in range(n_samples)]
            )
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
