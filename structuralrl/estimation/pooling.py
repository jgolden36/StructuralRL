"""Stage 2 — partial pooling theta_i = theta_0 + delta_i and lambda selection by CV.

The partial-pooling structure (Eq. pool) is implemented inside `second_step.penalized_pseudo_
likelihood` via the Gaussian penalty lambda * sum_i delta_i' Sigma^-1 delta_i. This module owns
the **selection of lambda**, which the notes are emphatic about: choose it by **cross-validation
across held-out campaigns**, scoring on fit to held-out interaction statistics / likelihood —
OUT of sample, so the pooling level is disciplined rather than chosen to flatter the estimates
(parent §"second step"). lambda -> inf is complete pooling (theta_i == theta_0); lambda -> 0 is
no pooling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .second_step import (
    SecondStepResult,
    held_out_loglik,
    penalized_pseudo_likelihood,
)


@dataclass
class PoolingCVResult:
    best_lambda: float
    lambdas: list[float]
    cv_scores: list[float]  # mean held-out log-likelihood per lambda (higher is better)
    refit: SecondStepResult  # refit on all folds at best_lambda


def select_lambda_cv(
    fold_data: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    psi_by_role: dict[str, np.ndarray],
    c_by_role: dict[str, np.ndarray],
    k: int,
    lambdas: list[float] | None = None,
    Sigma_inv: np.ndarray | None = None,
) -> PoolingCVResult:
    """Select lambda by leave-one-campaign-fold-out CV on held-out log-likelihood.

    Parameters
    ----------
    fold_data : list of folds; each fold is {role: (states, actions)} for the campaigns in it.
                Folds must partition by *campaign* so the score is genuinely out-of-campaign
                (never split a single campaign across train/test).
    psi_by_role, c_by_role : forward-simulated quantities (held fixed across folds; they depend on
                sigma-hat/F-hat, estimated once on the full first-step sample).
    """
    lambdas = lambdas or [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    roles = tuple(psi_by_role.keys())
    scores: list[float] = []

    for lam in lambdas:
        fold_scores = []
        for test_idx in range(len(fold_data)):
            train = _merge_folds([f for j, f in enumerate(fold_data) if j != test_idx], roles)
            test = fold_data[test_idx]
            res = penalized_pseudo_likelihood(train, psi_by_role, c_by_role, k, lam, Sigma_inv)
            fold_scores.append(held_out_loglik(res, test, psi_by_role, c_by_role))
        scores.append(float(np.mean(fold_scores)))

    best = int(np.argmax(scores))
    best_lambda = lambdas[best]
    all_data = _merge_folds(fold_data, roles)
    refit = penalized_pseudo_likelihood(all_data, psi_by_role, c_by_role, k, best_lambda, Sigma_inv)
    return PoolingCVResult(best_lambda, lambdas, scores, refit)


def _merge_folds(folds, roles) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {r: ([], []) for r in roles}
    for f in folds:
        for r in roles:
            if r in f:
                out[r][0].append(np.asarray(f[r][0], int))
                out[r][1].append(np.asarray(f[r][1], int))
    return {
        r: (
            np.concatenate(out[r][0]) if out[r][0] else np.array([], int),
            np.concatenate(out[r][1]) if out[r][1] else np.array([], int),
        )
        for r in roles
    }
