"""Example 04 — Bayesian Neural Network with SGLD and predictive uncertainty.

We fit a small 1-D regression task (a sine wave with heteroscedastic noise)
using a BayesianMLP trained with SGLD.  Outside the training interval the
epistemic uncertainty (due to weight uncertainty) should grow — the key
benefit of BNNs over point-estimate networks.
"""

from __future__ import annotations

import math

import torch

from pgm_handbook.deep_bayes.bnn import BayesianMLP, GaussianPrior
from pgm_handbook.deep_bayes.sgld import ensemble_predict, sgld_posterior_samples
from pgm_handbook.deep_bayes.dynamic_bnn import DynamicBNN


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_regression_data(n: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.empty(n, 1).uniform_(-3.0, 3.0)
    y = torch.sin(x) + 0.1 * torch.randn(n, 1)
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Bayesian MLP + SGLD — sine-wave regression with uncertainty")
    print("=" * 70)

    torch.manual_seed(0)
    n = 128
    noise_std = 0.1
    x_train, y_train = make_regression_data(n)

    # Small Bayesian MLP: 1 -> 16 -> 16 -> 1.  We run SGLD on the weight
    # *means* (a point in weight space); the variational log-sigma parameters
    # are frozen and unused because SGLD itself supplies the posterior spread.
    model = BayesianMLP(1, [16, 16], 1, prior=GaussianPrior(mean=0.0, std=1.0))
    mu_params = [p for k, p in model.named_parameters() if k.endswith("_mu")]
    for k, p in model.named_parameters():
        if not k.endswith("_mu"):
            p.requires_grad_(False)
    print("Trainable (SGLD) parameters:", sum(p.numel() for p in mu_params))

    def neg_log_lik(idx=None):
        """-log p(D | w), minibatch-rescaled to the full data set size."""
        xb, yb = (x_train, y_train) if idx is None else (x_train[idx], y_train[idx])
        resid = model.forward_mean(xb) - yb
        return 0.5 * (resid ** 2).sum() / noise_std ** 2 * (n / xb.size(0))

    # 1) MAP warm-up with Adam (burn-in shortcut for SGLD).
    adam = torch.optim.Adam(mu_params, lr=1e-2)
    for _ in range(1500):
        adam.zero_grad()
        prior_nll = 0.5 * sum((p ** 2).sum() for p in mu_params)
        ((neg_log_lik() + prior_nll) / n).backward()
        adam.step()
    with torch.no_grad():
        map_mse = float(((model.forward_mean(x_train) - y_train) ** 2).mean())
    print(f"  MAP train MSE = {map_mse:.4f} (noise var = {noise_std**2:.4f})")

    # 2) SGLD sampling from the posterior around the MAP.
    def minibatch_nll():
        return neg_log_lik(torch.randperm(n)[:32])

    snapshots = sgld_posterior_samples(
        mu_params, minibatch_nll, n_steps=6000, lr=3e-6,
        prior_precision=1.0, burn_in=2000, thin=20,
    )
    print(f"  Collected {len(snapshots)} SGLD posterior samples")

    # ------------------------------------------------------------------
    # Predictive uncertainty
    # ------------------------------------------------------------------
    print()
    print("Predictive uncertainty (posterior mean, epistemic and aleatoric std)")
    print("-" * 70)
    x_test = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [5.0], [8.0]]
    )
    out = ensemble_predict(model.forward_mean, mu_params, snapshots, x_test, noise_std)
    mean = out["mean"].squeeze(-1).numpy()
    epi = out["epistemic_std"].squeeze(-1).numpy()
    ale = out["aleatoric_std"].squeeze(-1).numpy()
    xs = x_test.squeeze(-1).tolist()
    for i, x in enumerate(xs):
        print(f"  x = {x:+.1f}  |  μ = {mean[i]:+.3f}"
              f"  σ_epi = {epi[i]:.3f}  σ_ale = {ale[i]:.3f}")

    inside = epi[xs.index(0.0)]
    outside = epi[xs.index(8.0)]
    print(f"\n  σ_epi(x=0) = {inside:.4f}   σ_epi(x=8) = {outside:.4f}")
    assert outside > inside, "BNN should be more uncertain outside the data"
    print("✓ Epistemic uncertainty grows in the extrapolation region")

    # Accuracy and coverage on the training interval (true function known).
    grid = torch.linspace(-3, 3, 300).unsqueeze(-1)
    out = ensemble_predict(model.forward_mean, mu_params, snapshots, grid, noise_std)
    rmse = float(((out["mean"] - torch.sin(grid)) ** 2).mean().sqrt())
    z = (torch.sin(grid) - out["mean"]) / out["total_std"]
    frac_95 = float((z.abs() <= 1.96).float().mean())
    print(f"\n  RMSE of posterior mean vs sin(x) on [-3, 3] = {rmse:.3f}")
    print(f"  Fraction of sin(x) within ±1.96σ_total        = {frac_95:.3f}")
    assert rmse < 0.15
    print("✓ Posterior mean recovers the sine wave")

    # ------------------------------------------------------------------
    # Dynamic BNN — toy sequential smoke-test
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("DynamicBNN — sequence smoke-test")
    print("=" * 70)
    dyn = DynamicBNN(in_features=2, hidden_size=8, out_features=1)
    x_seq = torch.randn(4, 5, 2)  # (batch=4, seq_len=5, in=2)
    y_seq = dyn(x_seq)
    print(f"  Input  shape: {tuple(x_seq.shape)}")
    print(f"  Output shape: {tuple(y_seq.shape)}")
    assert y_seq.shape == (4, 5, 1)

    pred = dyn.predictive(x_seq, n_samples=20)
    for k, v in pred.items():
        print(f"  {k:15s} shape = {tuple(v.shape)}")
    print("✓ DynamicBNN forward + predictive works")


if __name__ == "__main__":
    main()
