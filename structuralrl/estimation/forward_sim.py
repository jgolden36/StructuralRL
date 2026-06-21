"""Stage 2 — forward-simulated discounted feature / value sums (parent Eq. value).

Given Stage-1 rivals/own CCPs sigma-hat and transition F-hat, the choice-specific value of role
i is the simulated discounted payoff sum (Eq. value):

    V_i(sigma_i', sigma_-i; theta) = E[ sum_t beta^t pi_i(s_t, a_t; theta) | a_i ~ sigma_i', F ].

Because the payoff is linear in theta, V_i is linear in theta through forward-simulated
discounted **feature** sums, so the second step becomes a penalized logit (parent §"second
step"). This module computes those feature sums two ways:

* :func:`tabular_feature_sums` — EXACT linear-algebra solve on a finite grid (Phase 1 `sim`).
  Returns psi(s, a) (the discounted feature sum of taking a then following sigma_i) and the
  theta-independent entropy offset c(s, a), so that the implied logit is
      sigma_i(a | s; theta) = softmax_a [ psi(s, a) . theta + c(s, a) ].
  At the true theta with sigma_i = sigma_true this reproduces sigma_true exactly (Hotz-Miller),
  which is what makes the recovery test pass.
* :func:`monte_carlo_feature_sums` — general forward simulation against arbitrary sigma_-i / F,
  the route used when the state space is too large to solve exactly (and the high-variance regime
  the inequality estimator is meant for).

EULER_GAMMA enters the entropy offset; it is constant across actions within a state and so
cancels in the softmax — kept only for value-level correctness.
"""
from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329


def tabular_feature_sums(
    sigma: np.ndarray,
    P: np.ndarray,
    X: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact discounted feature sums psi(s,a) and entropy offset c(s,a) under own policy `sigma`.

    Parameters
    ----------
    sigma : (S, A) own CCP sigma_i(a | s) (the continuation policy is fixed at this, BBL-style).
    P : (S, A, S) transition F(s' | s, a).
    X : (S, A, k) structural features x(s, a).
    beta : discount factor.

    Returns
    -------
    psi : (S, A, k) discounted feature sum of taking a in s then following `sigma`.
    c   : (S, A) theta-independent discounted entropy/shock offset.
    """
    S, A, k = X.shape
    T = np.einsum("sa,sat->st", sigma, P)  # (S, S) policy-induced transition
    Minv = np.linalg.inv(np.eye(S) - beta * T)

    # state-level discounted feature occupancy: Psi = (I - beta T)^-1 Xbar
    Xbar = np.einsum("sa,sak->sk", sigma, X)  # (S, k)
    Psi_state = Minv @ Xbar  # (S, k)
    psi = X + beta * np.einsum("sat,tk->sak", P, Psi_state)  # (S, A, k)

    # state-level discounted entropy/shock stream: E = (I - beta T)^-1 Ebar
    # Ebar(s) = sum_a sigma(a|s) * (gamma - log sigma(a|s))  = expected shock given choice
    with np.errstate(divide="ignore"):
        shock = EULER_GAMMA - np.log(sigma)
    Ebar = np.einsum("sa,sa->s", sigma, np.where(sigma > 0, shock, 0.0))  # (S,)
    E_state = Minv @ Ebar  # (S,)
    c = beta * np.einsum("sat,t->sa", P, E_state)  # (S, A)
    return psi, c


def monte_carlo_feature_sums(
    sigma_own: np.ndarray,
    sigma_rivals,
    transition,
    X: np.ndarray,
    beta: float,
    horizon: int,
    n_paths: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """General forward-simulated feature sums by Monte Carlo (large / non-tabular state spaces).

    Mirrors :func:`tabular_feature_sums` but estimates psi(s,a) and c(s,a) by rolling out
    `n_paths` truncated trajectories of length `horizon` from each (s, a). Variance grows with
    horizon — the regime where the parent note prefers the inequality (BBL) second step and
    reports identified sets. `transition` exposes `.prob(s, a)` (a length-S vector) for the
    tabular case; replace with a surrogate sampler for real F-hat.
    """
    S, A, k = X.shape
    rng = np.random.default_rng(seed)
    psi = np.zeros((S, A, k))
    c = np.zeros((S, A))
    with np.errstate(divide="ignore"):
        shock = EULER_GAMMA - np.log(sigma_own)
    shock = np.where(sigma_own > 0, shock, 0.0)

    for s0 in range(S):
        for a0 in range(A):
            feat = np.zeros(k)
            ent = 0.0
            for _ in range(n_paths):
                s, a = s0, a0
                disc = 1.0
                feat += disc * X[s, a]
                for t in range(1, horizon):
                    s = int(rng.choice(S, p=transition.prob(s, a)))
                    a = int(rng.choice(A, p=sigma_own[s]))
                    disc *= beta
                    feat += disc * X[s, a]
                    ent += disc * shock[s, a]
            psi[s0, a0] = feat / n_paths
            c[s0, a0] = beta * ent / n_paths  # discounted future entropy (a-dependent via path)
    return psi, c
