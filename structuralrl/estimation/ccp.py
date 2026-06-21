"""Stage 1 (first step) — role-conditional conditional choice probabilities sigma_i(a | s).

Read structurally, fitting sigma_i is behavioral cloning; the structural content comes in Stage
2 when these CCPs are inverted (Hotz-Miller) into choice-specific value differences. Estimating
sigma_i here is an ASSUMPTION-light step: it conditions on whatever equilibrium generated the
data (the BBL convenience — no game solve, no equilibrium-selection stance at estimation time).

Two estimators:

* :class:`TabularCCP` — smoothed empirical frequencies over a finite (s, a) grid (Phase 1 `sim`).
  Exact in the data limit, which is what the recovery unit test needs.
* :class:`LogitCCP` — multinomial logit on phi(s); the real-data default. (NN variant lives with
  the encoder under the `nn` extra.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularCCP:
    """Laplace-smoothed empirical CCP over a finite discrete state-action grid.

    `alpha` is the Dirichlet/Laplace pseudocount; alpha -> 0 gives the raw MLE. Smoothing keeps
    log-CCP finite (the Hotz-Miller inversion needs log sigma) on states with thin counts.
    """

    n_states: int
    n_actions: int
    alpha: float = 1e-3

    def fit(self, states: np.ndarray, actions: np.ndarray) -> "TabularCCP":
        counts = np.full((self.n_states, self.n_actions), self.alpha, dtype=float)
        np.add.at(counts, (np.asarray(states, int), np.asarray(actions, int)), 1.0)
        self.sigma_ = counts / counts.sum(axis=1, keepdims=True)
        self.counts_ = counts
        return self

    @property
    def sigma(self) -> np.ndarray:
        return self.sigma_

    def log_sigma(self) -> np.ndarray:
        return np.log(self.sigma_)


def fit_role_ccps(
    sa_by_role: dict[str, tuple[np.ndarray, np.ndarray]],
    n_states: int,
    n_actions: int,
    alpha: float = 1e-3,
) -> dict[str, TabularCCP]:
    """Fit one :class:`TabularCCP` per role from {role: (states, actions)}."""
    return {
        role: TabularCCP(n_states, n_actions, alpha).fit(s, a)
        for role, (s, a) in sa_by_role.items()
    }


@dataclass
class LogitCCP:
    """Multinomial-logit CCP sigma_i(a | phi(s)) for real corpora (scaffold).

    Fits a linear softmax over phi(s) by L-BFGS. Provided so real-data Stage 1 has a concrete
    default; the recovery track uses :class:`TabularCCP`. For high-capacity phi use the torch
    encoder + a small head (the `nn` extra).
    """

    n_actions: int
    l2: float = 1e-4

    def fit(self, phi: np.ndarray, actions: np.ndarray) -> "LogitCCP":
        from scipy.optimize import minimize

        phi = np.asarray(phi, dtype=float)
        y = np.asarray(actions, dtype=int)
        n, d = phi.shape
        k = self.n_actions

        def negll(w):
            W = w.reshape(k, d)
            logits = phi @ W.T
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            ll = logp[np.arange(n), y].mean()
            return -ll + self.l2 * np.sum(W * W)

        res = minimize(negll, np.zeros(k * d), method="L-BFGS-B")
        self.W_ = res.x.reshape(k, d)
        return self

    def predict_log_proba(self, phi: np.ndarray) -> np.ndarray:
        logits = np.asarray(phi, float) @ self.W_.T
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
