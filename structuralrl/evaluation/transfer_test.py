"""Transfer of the pooled objective theta_0 (companion §5; CLAUDE.md §5).

Empirical content of "there is a common discovery objective to recover at all": recover payoffs
on one set of campaigns/labs and test whether they predict coordination on HELD-OUT ones, and
whether **theta_0** (the pooled component) is the part that transfers.

Heterogeneity across labs is large — never naively pool labs that played different equilibria;
use finite mixtures or estimate within homogeneous strata (CLAUDE.md §5/§7). This module tests
transfer; detecting/handling heterogeneity (mixtures) is a separate Phase-2 concern flagged here.

The test compares, on held-out campaigns, the predictive log-likelihood of:
  * the full recovered theta_i (theta_0 + delta_i), vs.
  * the pooled-only theta_0 (delta_i set to 0).
If theta_0 alone transfers about as well as the full payoff, the pooled objective is the
transferable part (the desired outcome). A large drop to pooled-only means role-specific
deviations are campaign-specific, i.e. weak transfer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimation.second_step import (
    SecondStepResult,
    held_out_loglik,
)


@dataclass
class TransferResult:
    loglik_full: float  # held-out loglik using theta_0 + delta_i
    loglik_pooled_only: float  # held-out loglik using theta_0 (delta_i = 0)
    loglik_role_shuffled: float  # control: theta_0 + delta_{wrong role}
    transfers: bool  # theta_0 transfers (pooled-only close to full, and beats shuffled control)
    margin: float

    def describe(self) -> str:
        v = "transfers" if self.transfers else "does NOT transfer"
        return (
            f"theta_0 {v}: full={self.loglik_full:.4f}, pooled_only={self.loglik_pooled_only:.4f}, "
            f"shuffled={self.loglik_role_shuffled:.4f}"
        )


def transfer_test(
    result: SecondStepResult,
    heldout_data: dict[str, tuple[np.ndarray, np.ndarray]],
    psi_by_role: dict[str, np.ndarray],
    c_by_role: dict[str, np.ndarray],
    tol: float = 0.05,
) -> TransferResult:
    """Test whether theta_0 predicts coordination on held-out campaigns.

    `tol` is the allowed held-out loglik drop from full to pooled-only for "transfers" to hold;
    pooled-only must also beat a role-shuffled control (else any constant would 'transfer').
    """
    full = held_out_loglik(result, heldout_data, psi_by_role, c_by_role)

    pooled_only = SecondStepResult(
        theta0=result.theta0,
        deltas={r: np.zeros_like(result.theta0) for r in result.roles},
        roles=result.roles,
        lam=result.lam,
        neg_loglik=result.neg_loglik,
        success=result.success,
        normalization=result.normalization,
    )
    ll_pooled = held_out_loglik(pooled_only, heldout_data, psi_by_role, c_by_role)

    shuffled = _role_shuffle(result)
    ll_shuffle = held_out_loglik(shuffled, heldout_data, psi_by_role, c_by_role)

    transfers = (full - ll_pooled <= tol) and (ll_pooled > ll_shuffle)
    return TransferResult(full, ll_pooled, ll_shuffle, transfers, full - ll_pooled)


def _role_shuffle(result: SecondStepResult) -> SecondStepResult:
    roles = list(result.roles)
    rotated = roles[1:] + roles[:1]
    deltas = {r: result.deltas[rotated[i]] for i, r in enumerate(roles)}
    return SecondStepResult(
        theta0=result.theta0,
        deltas=deltas,
        roles=result.roles,
        lam=result.lam,
        neg_loglik=result.neg_loglik,
        success=result.success,
        normalization=result.normalization,
    )
