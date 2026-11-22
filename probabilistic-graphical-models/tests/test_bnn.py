"""Tests for the Bayesian Neural Network, SGLD optimizer and Dynamic BNN."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from pgm_handbook.deep_bayes.bnn import (
    BayesianLinear,
    BayesianMLP,
    GaussianPrior,
)
from pgm_handbook.deep_bayes.sgld import SGLD
from pgm_handbook.deep_bayes.dynamic_bnn import DynamicBNN


# ---------------------------------------------------------------------------
# BayesianLinear
# ---------------------------------------------------------------------------
def test_bayesian_linear_shapes():
    layer = BayesianLinear(3, 5)
    x = torch.randn(7, 3)
    y = layer(x)
    assert y.shape == (7, 5)
    y_mean = layer.forward_mean(x)
    assert y_mean.shape == (7, 5)


def test_bayesian_linear_stochastic_samples_differ():
    # The forward pass should be stochastic: two calls yield different outputs.
    torch.manual_seed(0)
    layer = BayesianLinear(4, 4)
    # Make the posterior non-degenerate.
    with torch.no_grad():
        layer.weight_log_sigma.fill_(-1.0)
        layer.bias_log_sigma.fill_(-1.0)
    x = torch.randn(1, 4)
    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_kl_divergence_nonnegative():
    torch.manual_seed(0)
    layer = BayesianLinear(5, 3)
    kl = layer.kl_divergence()
    assert float(kl.detach()) >= 0.0


def test_kl_divergence_zero_at_prior():
    # Set q exactly to N(0, 1) to get KL = 0.
    layer = BayesianLinear(3, 2, GaussianPrior(0.0, 1.0), GaussianPrior(0.0, 1.0))
    with torch.no_grad():
        layer.weight_mu.fill_(0.0)
        layer.weight_log_sigma.fill_(0.0)
        layer.bias_mu.fill_(0.0)
        layer.bias_log_sigma.fill_(0.0)
    kl = float(layer.kl_divergence().detach())
    assert abs(kl) < 1e-6


# ---------------------------------------------------------------------------
# BayesianMLP predictive
# ---------------------------------------------------------------------------
def test_bayesian_mlp_predictive_shapes_and_keys():
    torch.manual_seed(0)
    model = BayesianMLP(2, [8, 8], 1)
    x = torch.randn(11, 2)
    out = model.predictive(x, n_samples=30)
    for k in ("mean", "epistemic_std", "aleatoric_std", "total_std"):
        assert k in out
        assert out[k].shape == (11, 1)


def test_bayesian_mlp_epistemic_decomposition():
    # total_var should equal epistemic_var + aleatoric_var (in std^2 form).
    torch.manual_seed(0)
    model = BayesianMLP(1, [4], 1)
    x = torch.randn(10, 1)
    out = model.predictive(x, n_samples=20)
    total_var = out["total_std"] ** 2
    epi_var = out["epistemic_std"] ** 2
    ale_var = out["aleatoric_std"] ** 2
    assert torch.allclose(total_var, epi_var + ale_var, atol=1e-6)


# ---------------------------------------------------------------------------
# SGLD
# ---------------------------------------------------------------------------
def test_sgld_optimizer_steps():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(5))
    opt = SGLD([p], lr=1e-2, prior_precision=0.0)

    # Zero gradient + prior → pure noise, variance ~ lr.
    opt.zero_grad()
    p.grad = torch.zeros_like(p)
    before = p.detach().clone()
    opt.step()
    delta = (p.detach() - before).numpy()
    # Variance of delta should be close to lr.
    empirical_var = float((delta ** 2).mean())
    assert 0.3 * 1e-2 < empirical_var < 3.0 * 1e-2


def test_sgld_converges_to_posterior_mean():
    # Simple 1-D quadratic loss: L(w) = (w - 3)^2 / 2  → posterior mean = 3.
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([0.0]))
    opt = SGLD([w], lr=1e-2, prior_precision=0.0)

    samples = []
    n_steps = 2000
    for t in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = 1e-1 * (1.0 + t) ** -0.6
        opt.zero_grad()
        loss = 0.5 * (w - 3.0) ** 2
        loss.backward()
        opt.step()
        if t > n_steps // 2:
            samples.append(w.item())
    mean = float(sum(samples) / len(samples))
    assert abs(mean - 3.0) < 0.3  # posterior is N(3, 1); MC error ~ 0.1


# ---------------------------------------------------------------------------
# DynamicBNN
# ---------------------------------------------------------------------------
def test_dynamic_bnn_shapes():
    torch.manual_seed(0)
    dyn = DynamicBNN(in_features=2, hidden_size=6, out_features=3)
    x = torch.randn(4, 5, 2)  # (B, T, in)
    y = dyn(x)
    assert y.shape == (4, 5, 3)


def test_dynamic_bnn_predictive_keys():
    torch.manual_seed(0)
    dyn = DynamicBNN(in_features=2, hidden_size=4, out_features=1)
    x = torch.randn(2, 3, 2)
    out = dyn.predictive(x, n_samples=15)
    for k in ("mean", "epistemic_std", "aleatoric_std", "total_std"):
        assert out[k].shape == (2, 3, 1)


def test_dynamic_bnn_kl_nonnegative():
    torch.manual_seed(0)
    dyn = DynamicBNN(2, 4, 1)
    assert float(dyn.kl_divergence().detach()) >= 0.0


# ---------------------------------------------------------------------------
# BayesianMLP.forward_mean + SGLD helpers
# ---------------------------------------------------------------------------
def test_mlp_forward_mean_is_deterministic():
    torch.manual_seed(0)
    mlp = BayesianMLP(2, [4], 1)
    x = torch.randn(5, 2)
    assert torch.equal(mlp.forward_mean(x), mlp.forward_mean(x))
    assert mlp.forward_mean(x).shape == (5, 1)


def test_sgld_posterior_samples_recovers_gaussian_posterior():
    from pgm_handbook.deep_bayes.sgld import sgld_posterior_samples

    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(1))
    # posterior ∝ exp(-(w-3)^2/2) (no prior): N(3, 1)
    snaps = sgld_posterior_samples(
        [w], lambda: 0.5 * ((w - 3.0) ** 2).sum(), n_steps=6000, lr=5e-2,
        prior_precision=0.0, burn_in=1000, thin=5,
    )
    vals = torch.stack([s[0] for s in snaps]).squeeze()
    assert len(snaps) == 1000
    assert abs(float(vals.mean()) - 3.0) < 0.2
    assert abs(float(vals.std()) - 1.0) < 0.25


def test_ensemble_predict_restores_params_and_decomposes_variance():
    from pgm_handbook.deep_bayes.sgld import ensemble_predict

    w = torch.nn.Parameter(torch.tensor([1.0]))
    snaps = [[torch.tensor([0.0])], [torch.tensor([2.0])]]
    x = torch.tensor([[1.0]])
    out = ensemble_predict(lambda t: t * w, [w], snaps, x, noise_std=0.5)
    assert float(w.detach()) == 1.0                         # restored
    assert math.isclose(float(out["mean"]), 1.0)
    assert math.isclose(float(out["epistemic_std"]), 1.0)   # values 0 and 2
    assert math.isclose(float(out["aleatoric_std"]), 0.5)
    assert math.isclose(float(out["total_std"]), math.sqrt(1.25), rel_tol=1e-6)


@pytest.mark.parametrize("mode", [True, False])
def test_predictive_preserves_training_mode(mode):
    from pgm_handbook.deep_bayes.dynamic_bnn import DynamicBNN

    mlp = BayesianMLP(1, [4], 1)
    mlp.train(mode)
    mlp.predictive(torch.zeros(2, 1), n_samples=3)
    assert mlp.training is mode

    dyn = DynamicBNN(1, 4, 1)
    dyn.train(mode)
    dyn.predictive(torch.zeros(2, 3, 1), n_samples=3)
    assert dyn.training is mode
