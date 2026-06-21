"""Stage 1 (first step) — transition model F-hat(s' | s, a); surrogate / digital-twin adapter.

F is **partly deterministic** (queue/accept moves advance the campaign mechanically) and
**partly stochastic** (nature's response to an experiment: whether a synthesis succeeds, what
property is measured). Where a validated surrogate / digital twin exists it supplies F-hat AND
the Stage-4 rollout environment (companion note). The surrogate is reliable only near the
explored region — flag rollouts that leave its validated support (CLAUDE.md §7).

Estimators:

* :class:`TabularTransition` — smoothed empirical F-hat from (s, a, s') counts (Phase 1 `sim`).
* :class:`SurrogateTransition` — adapter wrapping an external simulator/digital twin, with a
  `support` check that marks off-support queries (the binding sim-to-real risk for Stage 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class Transition(Protocol):
    def prob(self, s: int, a: int) -> np.ndarray:  # pragma: no cover - protocol
        """Return P(s' | s, a) as a length-S vector."""
        ...


@dataclass
class TabularTransition:
    """Laplace-smoothed empirical transition tensor F-hat(s'|s,a) over a finite grid."""

    n_states: int
    n_actions: int
    alpha: float = 1e-3

    def fit(self, transitions) -> "TabularTransition":
        counts = np.full((self.n_states, self.n_actions, self.n_states), self.alpha, dtype=float)
        for s, a, s2 in transitions:
            counts[int(s), int(a), int(s2)] += 1.0
        self.P_ = counts / counts.sum(axis=2, keepdims=True)
        return self

    @property
    def P(self) -> np.ndarray:
        return self.P_

    def prob(self, s: int, a: int) -> np.ndarray:
        return self.P_[s, a]


@dataclass
class SurrogateTransition:
    """Adapter around a validated surrogate / digital twin (supplies F-hat and Stage-4 rollouts).

    `step_fn(s, a, rng) -> s'` is the simulator; `support_fn(s, a) -> bool` marks whether the
    query lies in the surrogate's validated region. Off-support queries are counted and exposed
    via `off_support_fraction` so Stage 4 can bound how far improvement travels from the human
    manifold.
    """

    step_fn: Callable
    support_fn: Callable[[int, int], bool] | None = None

    def __post_init__(self) -> None:
        self._n_queries = 0
        self._n_off_support = 0

    def step(self, s, a, rng: np.random.Generator):
        self._n_queries += 1
        if self.support_fn is not None and not self.support_fn(s, a):
            self._n_off_support += 1
        return self.step_fn(s, a, rng)

    @property
    def off_support_fraction(self) -> float:
        return 0.0 if self._n_queries == 0 else self._n_off_support / self._n_queries
