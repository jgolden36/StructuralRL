"""Stage 2 (second step) — partially pooled payoffs by penalized pseudo-likelihood (Eq. ppl).

Primary estimator (parent Eq. ppl). With sigma-hat and F-hat from Stage 1, the choice-specific
values are linear in theta through the forward-simulated feature sums psi(s,a) (forward_sim),
the implied policy is the logit sigma_i(a|s; theta) = softmax_a[psi(s,a).theta_i + c(s,a)], and
the partially pooled payoffs maximise

    sum_i sum_t log sigma_i(a_it | s_t; theta0 + delta_i)  -  lambda * sum_i delta_i' Sigma^-1 delta_i.

This is a **penalized multinomial logit** (convex in (theta0, {delta_i})) — no game solve, no
equilibrium re-computation. The penalty is the Gaussian partial-pooling prior of Eq. pool;
lambda is selected by cross-validation across held-out campaigns (see pooling.py), NOT in sample.

The robustness route (BBL inequality, Eq. pooled) lives in :func:`bbl_inequality_objective`; it
is **misspecified under quantal play** (bias grows with temperature), so callers report identified
SETS there (identified_set.py), not points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .normalization import Normalization


@dataclass
class SecondStepResult:
    """Recovered partially pooled payoffs and the pieces needed to report identified sets."""

    theta0: np.ndarray
    deltas: dict[str, np.ndarray]
    roles: tuple[str, ...]
    lam: float
    neg_loglik: float
    success: bool
    normalization: Normalization = field(default_factory=Normalization)
    hessian: np.ndarray | None = None  # of the negative penalized objective at the optimum

    def theta(self, role: str) -> np.ndarray:
        return self.theta0 + self.deltas[role]

    @property
    def theta_by_role(self) -> dict[str, np.ndarray]:
        return {r: self.theta(r) for r in self.roles}


def _pack(theta0, deltas, roles):
    return np.concatenate([theta0] + [deltas[r] for r in roles])


def _unpack(w, k, roles):
    theta0 = w[:k]
    deltas = {r: w[k + i * k : k + (i + 1) * k] for i, r in enumerate(roles)}
    return theta0, deltas


def penalized_pseudo_likelihood(
    data_by_role: dict[str, tuple[np.ndarray, np.ndarray]],
    psi_by_role: dict[str, np.ndarray],
    c_by_role: dict[str, np.ndarray],
    k: int,
    lam: float,
    Sigma_inv: np.ndarray | None = None,
    normalization: Normalization | None = None,
    w0: np.ndarray | None = None,
) -> SecondStepResult:
    """Maximise the penalized pseudo-likelihood (Eq. ppl). Returns a :class:`SecondStepResult`.

    Parameters
    ----------
    data_by_role : role -> (states, actions) integer arrays of observed (s, a).
    psi_by_role  : role -> (S, A, k) forward-simulated feature sums (forward_sim).
    c_by_role    : role -> (S, A) theta-independent logit offset (forward_sim).
    k            : payoff feature dimension.
    lam          : pooling strength lambda (>=0). lam -> inf is complete pooling, lam -> 0 none.
    Sigma_inv    : (k, k) prior precision Sigma_delta^-1 (defaults to identity).
    """
    roles = tuple(data_by_role.keys())
    Sigma_inv = np.eye(k) if Sigma_inv is None else np.asarray(Sigma_inv, float)
    norm = normalization or Normalization()

    # precompute per-role observation design: for each obs, the (A, k) psi and (A,) c at its state
    obs = {}
    for r in roles:
        s, a = np.asarray(data_by_role[r][0], int), np.asarray(data_by_role[r][1], int)
        obs[r] = (psi_by_role[r][s], c_by_role[r][s], a)  # (n,A,k), (n,A), (n,)

    def neg_obj_and_grad(w):
        theta0, deltas = _unpack(w, k, roles)
        f = 0.0
        g0 = np.zeros(k)
        gd = {r: np.zeros(k) for r in roles}
        for r in roles:
            psi_n, c_n, y = obs[r]  # (n,A,k),(n,A),(n,)
            theta_i = theta0 + deltas[r]
            logits = psi_n @ theta_i + c_n  # (n, A)
            logits -= logits.max(axis=1, keepdims=True)
            ex = np.exp(logits)
            p = ex / ex.sum(axis=1, keepdims=True)  # (n, A)
            n = len(y)
            logp_y = np.log(p[np.arange(n), y])
            f -= logp_y.sum()
            # grad of -loglik wrt theta_i: sum_t (E_p[psi] - psi_y)
            Ep = np.einsum("na,nak->nk", p, psi_n)  # (n, k)
            psi_y = psi_n[np.arange(n), y]  # (n, k)
            gi = (Ep - psi_y).sum(axis=0)
            g0 += gi
            gd[r] += gi
        # penalty lam * sum_i delta' Sigma_inv delta  (added to the NEGATIVE objective)
        for r in roles:
            d = deltas[r]
            f += lam * d @ Sigma_inv @ d
            gd[r] += 2.0 * lam * (Sigma_inv @ d)
        return f, _pack(g0, gd, roles)

    if w0 is None:
        w0 = np.zeros(k * (1 + len(roles)))
    res = minimize(neg_obj_and_grad, w0, jac=True, method="L-BFGS-B")
    theta0, deltas = _unpack(res.x, k, roles)

    # asymptotic Hessian (of the negative penalized objective) for identified-set reporting
    hess = _numeric_hessian(lambda w: neg_obj_and_grad(w)[0], res.x)

    return SecondStepResult(
        theta0=theta0,
        deltas=deltas,
        roles=roles,
        lam=lam,
        neg_loglik=float(res.fun),
        success=bool(res.success),
        normalization=norm,
        hessian=hess,
    )


def held_out_loglik(
    result: SecondStepResult,
    data_by_role: dict[str, tuple[np.ndarray, np.ndarray]],
    psi_by_role: dict[str, np.ndarray],
    c_by_role: dict[str, np.ndarray],
) -> float:
    """Mean per-observation log-likelihood of `result`'s payoffs on held-out (s, a) data.

    The out-of-sample score that lambda CV optimises (pooling.py) — scoring on held-out fit
    rather than in-sample, so the pooling level is not chosen to flatter the estimates.
    """
    total, count = 0.0, 0
    for r in data_by_role:
        s, a = np.asarray(data_by_role[r][0], int), np.asarray(data_by_role[r][1], int)
        theta_i = result.theta(r)
        logits = psi_by_role[r][s] @ theta_i + c_by_role[r][s]
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        total += logp[np.arange(len(a)), a].sum()
        count += len(a)
    return total / max(count, 1)


def bbl_inequality_objective(
    sigma_by_role: dict[str, np.ndarray],
    psi_by_role: dict[str, np.ndarray],
    c_by_role: dict[str, np.ndarray],
    theta0: np.ndarray,
    deltas: dict[str, np.ndarray],
    perturbations,
    lam: float = 0.0,
    Sigma_inv: np.ndarray | None = None,
) -> float:
    """BBL inequality objective Q_BBL (Eq. pooled) — ROBUSTNESS route, report SETS not points.

    Penalizes profitable one-shot deviations: for each sampled alternative policy sigma' from
    `perturbations`, the value gap max{0, V(sigma') - V(sigma_hat)} squared. Under quantal play
    the observed policy is not exactly optimal in the deterministic payoff, so this is
    misspecified at the truth (bias grows with temperature) — hence identified sets.

    `perturbations` is an iterable of (role, sigma_prime) with sigma_prime an (S, A) policy.
    Values use the linear form V_i(sigma') = sum_s d_sigma'(s) * (psi(s,.).theta + c) collapsed
    to the policy-weighted choice value; here approximated by the per-state expected
    choice-specific value under sigma'.
    """
    k = len(theta0)
    Sigma_inv = np.eye(k) if Sigma_inv is None else np.asarray(Sigma_inv, float)

    def value(role, sigma):
        theta_i = theta0 + deltas[role]
        v = psi_by_role[role] @ theta_i + c_by_role[role]  # (S, A) choice-specific value
        return np.einsum("sa,sa->s", sigma, v)  # (S,) expected value per state under sigma

    q = 0.0
    for role, sigma_prime in perturbations:
        base = value(role, sigma_by_role[role])
        alt = value(role, sigma_prime)
        gap = np.maximum(0.0, alt - base)
        q += float((gap ** 2).sum())
    for r in deltas:
        q += lam * float(deltas[r] @ Sigma_inv @ deltas[r])
    return q


def _numeric_hessian(f, x, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        xi = x.copy()
        xi[i] += eps
        for j in range(i, n):
            xij = xi.copy()
            xij[j] += eps
            xj = x.copy()
            xj[j] += eps
            H[i, j] = H[j, i] = (f(xij) - f(xi) - f(xj) + fx) / (eps * eps)
    return H
