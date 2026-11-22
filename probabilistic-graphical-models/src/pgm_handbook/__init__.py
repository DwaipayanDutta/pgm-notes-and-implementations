"""pgm_handbook: a hands-on handbook of probabilistic graphical models.

Subpackages
-----------
representation : Bayesian / Markov networks, local CPD structures, template models
inference      : exact and Monte-Carlo / MCMC inference
deep_bayes     : Bayesian neural networks, SGLD, dynamic BNN (requires PyTorch)
"""

__version__ = "0.1.0"

from . import representation, inference  # noqa: F401

try:  # torch is an optional heavy dependency
    from . import deep_bayes  # noqa: F401
except ImportError:  # pragma: no cover
    deep_bayes = None  # type: ignore[assignment]

__all__ = ["representation", "inference", "deep_bayes", "__version__"]
