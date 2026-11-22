"""Compact CPD representations.

All CPDs implement ``prob(value, parent_assignment)``.  Tabular CPDs also
expose a ``domain`` attribute; each class here does the same when possible.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Sequence, Tuple


class DeterministicCPD:
    """``X = f(Pa(X))`` with probability 1."""

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        func: Callable[..., object],
        domain: Sequence | None = None,
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.func = func
        self.domain: List = list(domain) if domain is not None else []

    def distribution(self, parent_assignment: Sequence) -> Dict:
        return {self.func(*parent_assignment): 1.0}

    def prob(self, value, parent_assignment: Sequence) -> float:
        return 1.0 if self.func(*parent_assignment) == value else 0.0


class TreeCPD:
    """Generic decision-tree CPD.

    ``tree`` maps an assignment of some prefix of ``parents`` to a
    distribution over ``variable``.  Lookup tries the longest matching
    prefix first (full assignment, then shorter prefixes down to ``()``);
    if none matches, the ``default`` distribution is used.
    """

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        tree: Mapping[Tuple, Mapping],
        default: Mapping | None = None,
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.tree = {tuple(k): dict(v) for k, v in tree.items()}
        self.default = dict(default) if default else {}
        dom = set(self.default.keys())
        for d in self.tree.values():
            dom.update(d.keys())
        self.domain: List = sorted(dom, key=str)

    def distribution(self, parent_assignment: Sequence) -> Dict:
        key = tuple(parent_assignment)
        for n in range(len(key), -1, -1):
            k = key[:n]
            if k in self.tree:
                return dict(self.tree[k])
        return dict(self.default)

    def prob(self, value, parent_assignment: Sequence) -> float:
        return self.distribution(parent_assignment).get(value, 0.0)


class MultiplexerCPD:
    """Multiplexer: child takes the value of the parent chosen by ``selector``.

    ``parents = (selector,) + tuple(choices)``; ``parent_assignment[0]``
    is an integer index into ``choices``.
    """

    def __init__(
        self,
        variable: str,
        selector: str,
        choices: Sequence[str],
        domain: Sequence | None = None,
    ) -> None:
        self.variable = variable
        self.selector = selector
        self.choices: Tuple[str, ...] = tuple(choices)
        self.parents: Tuple[str, ...] = (selector,) + self.choices
        self.domain: List = list(domain) if domain is not None else []

    def distribution(self, parent_assignment: Sequence) -> Dict:
        idx = int(parent_assignment[0])
        return {parent_assignment[1 + idx]: 1.0}

    def prob(self, value, parent_assignment: Sequence) -> float:
        idx = int(parent_assignment[0])
        return 1.0 if parent_assignment[1 + idx] == value else 0.0


class NoisyORCPD:
    r"""Noisy-OR: each active parent independently causes the child.

    ``q[i] = P(child=True | only parent i True, all others False)``.

    ``P(child=False | pa) = \prod_{i: X_i=True} (1 - q_i)``
    """

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        q: Sequence[float],
        leak: float = 0.0,
        domain: Sequence = (False, True),
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.q: List[float] = list(q)
        self.leak = leak
        self.domain: List = list(domain)

    def distribution(self, parent_assignment: Sequence) -> Dict:
        prod = 1.0 - self.leak
        for p, qi in zip(parent_assignment, self.q):
            if p:
                prod *= 1.0 - qi
        p_true = 1.0 - prod
        return {False: 1.0 - p_true, True: p_true}

    def prob(self, value, parent_assignment: Sequence) -> float:
        return self.distribution(parent_assignment)[bool(value)]


class NoisyMAXCPD:
    """Noisy-MAX: each active parent independently draws a level; child = max."""

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        distributions: Mapping[str, Mapping[int, float]],
        default: int = 0,
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.dists: Dict[str, Dict[int, float]] = {
            p: dict(d) for p, d in distributions.items()
        }
        self.default = default
        levels = {default}
        for d in self.dists.values():
            levels.update(d.keys())
        self.domain: List = sorted(levels)

    def distribution(self, parent_assignment: Sequence) -> Dict:
        active = [self.parents[i] for i, v in enumerate(parent_assignment) if v]
        if not active:
            return {self.default: 1.0}
        levels = self.domain
        cdf = {k: 1.0 for k in levels}
        for p in active:
            d = self.dists[p]
            running = 0.0
            step_cdf = {}
            for k in levels:
                running += d.get(k, 0.0)
                step_cdf[k] = running
            for k in levels:
                cdf[k] *= step_cdf[k]
        out, prev = {}, 0.0
        for k in levels:
            out[k] = cdf[k] - prev
            prev = cdf[k]
        return out

    def prob(self, value, parent_assignment: Sequence) -> float:
        return self.distribution(parent_assignment).get(value, 0.0)


class LogisticCPD:
    """Logistic (sigmoid) CPD: ``P(X=True|pa) = sigmoid(w·pa + b)``."""

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        weights: Sequence[float],
        bias: float = 0.0,
        domain: Sequence = (False, True),
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.weights: List[float] = list(weights)
        self.bias = bias
        self.domain: List = list(domain)

    def distribution(self, parent_assignment: Sequence) -> Dict:
        z = self.bias + sum(
            w * float(p) for w, p in zip(self.weights, parent_assignment)
        )
        p = 1.0 / (1.0 + math.exp(-z))
        return {False: 1.0 - p, True: p}

    def prob(self, value, parent_assignment: Sequence) -> float:
        return self.distribution(parent_assignment)[bool(value)]


class LinearGaussianCPD:
    """Linear-Gaussian: ``X ~ N(w·pa + b, sigma^2)``.

    Continuous CPDs do not expose a discrete ``domain`` and should be handled
    with sampling-based inference.
    """

    def __init__(
        self,
        variable: str,
        parents: Sequence[str],
        weights: Sequence[float],
        bias: float = 0.0,
        variance: float = 1.0,
    ) -> None:
        self.variable = variable
        self.parents: Tuple[str, ...] = tuple(parents)
        self.weights: List[float] = list(weights)
        self.bias = bias
        self.variance = variance

    def mean(self, parent_assignment: Sequence[float]) -> float:
        return self.bias + sum(w * p for w, p in zip(self.weights, parent_assignment))

    def pdf(self, value: float, parent_assignment: Sequence[float]) -> float:
        mu = self.mean(parent_assignment)
        return math.exp(-0.5 * (value - mu) ** 2 / self.variance) / math.sqrt(
            2.0 * math.pi * self.variance
        )

    def sample(self, parent_assignment: Sequence[float], rng) -> float:
        mu = self.mean(parent_assignment)
        return float(rng.normal(mu, math.sqrt(self.variance)))
