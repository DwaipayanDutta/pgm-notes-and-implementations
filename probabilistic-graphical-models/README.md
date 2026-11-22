# pgm-handbook

A compact, readable Python package for **probabilistic graphical models** — from representation and
exact inference to Monte-Carlo / MCMC and Bayesian deep learning. Built for learning: small pure-Python
implementations that you can step through, each verified against known results.

| Area | Module | What you get |
|---|---|---|
| Bayesian networks | `representation/bayesian_network.py` | DAG, `TabularCPD`, joint factorisation, d-separation |
| Markov networks | `representation/markov_network.py` | clique potentials, log-linear energy, exact / IS partition function |
| Local structure | `representation/local_structures.py` | Deterministic, Tree, Multiplexer, Noisy-OR, Noisy-MAX, Logistic, Linear-Gaussian CPDs |
| Template models | `representation/template_models.py` | `PlateModel` grounding, `DynamicBayesianNetwork` (2-slice → unrolled HMM) |
| Exact inference | `inference/exact.py` | enumeration, variable elimination |
| Sampling | `inference/{importance,inverse_transform,rejection}.py` | IS, self-normalised IS, inverse CDF (continuous + discrete), rejection (envelope + box) |
| MCMC | `inference/mcmc.py` | Metropolis–Hastings, Gibbs |
| Bayesian deep learning | `deep_bayes/` | `BayesianLinear/MLP`, SGLD optimiser + posterior-sample helpers, recurrent `DynamicBNN` |

## Install

```bash
pip install pgm-handbook            # numpy, networkx
pip install "pgm-handbook[deep]"    # + torch, for deep_bayes
```

Python ≥ 3.9. `import pgm_handbook` works without PyTorch; `pgm_handbook.deep_bayes` needs it.

## Quick start

```python
from pgm_handbook.representation import BayesianNetwork, TabularCPD
from pgm_handbook.inference import enumeration_ask, variable_elimination

bn = BayesianNetwork()
for n in ("B", "E", "A"):
    bn.add_node(n)
bn.add_edge("B", "A"); bn.add_edge("E", "A")
bn.set_cpd(TabularCPD("B", (), {(): {True: .001, False: .999}}))
bn.set_cpd(TabularCPD("E", (), {(): {True: .002, False: .998}}))
bn.set_cpd(TabularCPD("A", ("B", "E"), {
    (True, True): {True: .95, False: .05}, (True, False): {True: .94, False: .06},
    (False, True): {True: .29, False: .71}, (False, False): {True: .001, False: .999}}))

print(enumeration_ask("B", {"A": True}, bn))       # {False: 0.626, True: 0.374}
```

The full 5-node burglary network reproduces the textbook value
`P(Burglary | JohnCalls, MaryCalls) ≈ 0.284` (`examples/01_burglary_alarm_inference.py`).

## Examples and notebook

The examples and the demo notebook live in the
[GitHub repository](https://github.com/DwaipayanDutta/pgm-notes-and-implementations)
(they are also included in the source distribution). From a checkout:

```bash
python examples/01_burglary_alarm_inference.py   # exact inference, d-separation, explaining away
python examples/02_local_cpds_and_plates.py      # local CPDs, plate model, DBN
python examples/03_monte_carlo_and_mcmc.py       # IS, inverse transform, rejection, MH, Gibbs
python examples/04_bnn_sgld_uncertainty.py       # BNN + SGLD, epistemic vs aleatoric uncertainty
jupyter lab notebooks/pgm_handbook_demo.ipynb    # everything above, with plots
```

Each example asserts its numerical results, so a clean run means the numbers match theory.

## Development

```bash
git clone https://github.com/DwaipayanDutta/pgm-notes-and-implementations
cd pgm-notes-and-implementations/probabilistic-graphical-models
pip install -e ".[dev]"   # + pytest, torch, scipy, matplotlib, jupyter
pytest                    # unit tests, example smoke tests, notebook execution
```

Torch-dependent tests and examples are skipped automatically when PyTorch is not installed.

## API notes

* **Rejection sampling.** `rejection_sample(target, proposal, draw, M, n)` requires the envelope
  `target(x) <= M * proposal(x)`. `rejection_sample_box(target, lower, upper, M, n)` uses a uniform
  proposal on the box, and `M` must bound `target` itself on the box. Both stop with a
  `RuntimeError` after `max_proposals` draws (default `max(10_000, 1000 * n)`) rather than looping
  forever on a bad envelope.
* **SGLD.** `sgld_posterior_samples(params, neg_log_lik, ...)` expects the minibatch negative
  log-likelihood already rescaled to the full data-set size (`N/B * sum`); the Gaussian prior is
  supplied through `prior_precision`. `ensemble_predict` turns the collected samples into a predictive
  distribution with `mean`, `epistemic_std`, `aleatoric_std` and `total_std`.
* **Scaling.** Exact inference is exponential in the number of variables by design — it is a teaching
  implementation, intended for small networks.

## Layout

```
src/pgm_handbook/{representation,inference,deep_bayes}/
examples/    four runnable scripts
notebooks/   pgm_handbook_demo.ipynb (executed, outputs included)
tests/       pytest suite
```

## License

MIT
