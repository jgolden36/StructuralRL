"""State-sufficiency falsification test (companion §5; CLAUDE.md §5).

Assumption (state sufficiency): phi(s) is a sufficient statistic for continuation play. We
FALSIFY it: train a next-move predictor on phi(s) vs. phi(s) augmented with *dropped context*
(free-text rationale / stated hypothesis / prior-campaign memory). A significant predictive gain
from the dropped context is **evidence AGAINST sufficiency** — and then the binding constraint is
*instrumentation* (log the rationale), not modeling.

This is a gating test, not a fit metric: the go/no-go rule requires it to NOT reject. We report a
held-out log-likelihood gain and a likelihood-ratio test p-value; the caller compares against a
pre-registered alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimation.ccp import LogitCCP


@dataclass
class SufficiencyResult:
    loglik_base: float  # held-out mean loglik on phi(s)
    loglik_augmented: float  # held-out mean loglik on phi(s) + dropped context
    loglik_gain: float
    lr_stat: float
    p_value: float
    rejects_sufficiency: bool  # True => evidence AGAINST sufficiency (BAD for go/no-go)

    def describe(self) -> str:
        verdict = "REJECTS sufficiency" if self.rejects_sufficiency else "does not reject"
        return f"sufficiency test {verdict}: gain={self.loglik_gain:.4f}, p={self.p_value:.4g}"


def sufficiency_test(
    phi_train: np.ndarray,
    context_train: np.ndarray,
    actions_train: np.ndarray,
    phi_test: np.ndarray,
    context_test: np.ndarray,
    actions_test: np.ndarray,
    n_actions: int,
    alpha: float = 0.05,
) -> SufficiencyResult:
    """Compare next-move predictors on phi vs [phi | dropped-context], held out.

    `context_*` are the encoded dropped-context features (e.g. rationale embeddings). The
    likelihood-ratio statistic uses the in-sample improvement with df = (#extra params); the
    decision uses the held-out gain sign together with the LR p-value, so a context that only
    overfits in sample does not trip the test.
    """
    from scipy.stats import chi2

    base = LogitCCP(n_actions).fit(phi_train, actions_train)
    aug_train = np.hstack([phi_train, context_train])
    aug_test = np.hstack([phi_test, context_test])
    aug = LogitCCP(n_actions).fit(aug_train, actions_train)

    ll_base = _mean_loglik(base, phi_test, actions_test)
    ll_aug = _mean_loglik(aug, aug_test, actions_test)
    gain = ll_aug - ll_base

    # LR statistic on the TRAINING fit (asymptotic chi-square), df = extra params added.
    ll_base_tr = _mean_loglik(base, phi_train, actions_train) * len(actions_train)
    ll_aug_tr = _mean_loglik(aug, aug_train, actions_train) * len(actions_train)
    lr = 2.0 * max(ll_aug_tr - ll_base_tr, 0.0)
    df = n_actions * context_train.shape[1]
    p = float(chi2.sf(lr, df)) if df > 0 else 1.0

    rejects = (gain > 0) and (p < alpha)
    return SufficiencyResult(ll_base, ll_aug, gain, lr, p, rejects)


def _mean_loglik(model: LogitCCP, phi: np.ndarray, y: np.ndarray) -> float:
    logp = model.predict_log_proba(phi)
    return float(logp[np.arange(len(y)), np.asarray(y, int)].mean())
