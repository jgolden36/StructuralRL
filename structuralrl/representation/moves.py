"""Stage 0 — the discovery move taxonomy M and the functional roles.

The taxonomy is fixed in advance (companion note §2): treating contributions at the level of
*strategic moves* is the bridge from combinatorial text/protocol space to the small discrete
action set the structural estimators assume. The realized content of a move (a specific
protocol/compound) is a *within-move structured choice* carried in `EventRecord.payload`, not a
separate move.

`validate_taxonomy` implements the paper's requirement that the taxonomy be *validated* by
predicting held-out moves and checked for sensitivity to granularity — it is a check, not an
assumption: a taxonomy that does not predict held-out behavior is the wrong taxonomy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Functional roles (companion note §2 "Players"): the players of the DMTA game.
ROLES: tuple[str, ...] = ("proposer", "designer", "executor", "analyst", "arbiter")

# Discovery-loop move taxonomy M (companion note §2 "Actions" / Stage 0).
MOVES: tuple[str, ...] = (
    "propose",
    "design",
    "execute",
    "characterize",
    "accept",
    "reject",
    "replicate",
    "pivot",
    "defer",
    "escalate",
    "stop",
)

# The canonical DMTA cadence used by interaction-stat tests (proposer -> ... -> arbiter).
DMTA_ROLE_ORDER: tuple[str, ...] = ("proposer", "designer", "executor", "analyst", "arbiter")

# Pre-registered diagnostic events / responses (companion §5.4, evaluation).
PREREGISTERED_EVENTS: tuple[str, ...] = (
    "anomalous_measurement",
    "failed_synthesis",
    "replication_failure",
    "contradictory_result",
    "budget_threshold_crossing",
)
DIAGNOSTIC_RESPONSES: tuple[str, ...] = ("replicate", "escalate", "pivot", "stop")


def move_index(moves: Sequence[str] = MOVES) -> dict[str, int]:
    """Map move name -> integer action id (stable ordering of `moves`)."""
    return {m: i for i, m in enumerate(moves)}


def role_index(roles: Sequence[str] = ROLES) -> dict[str, int]:
    return {r: i for i, r in enumerate(roles)}


def validate_taxonomy(
    moves_observed: Sequence[str],
    predicted_logits: np.ndarray,
    moves: Sequence[str] = MOVES,
) -> dict[str, float]:
    """Validate the taxonomy by held-out move prediction (Stage 0 validation, NOT assumed).

    Parameters
    ----------
    moves_observed : held-out realized move names, length n.
    predicted_logits : (n, |M|) scores from a next-move predictor on phi(s).
    moves : the taxonomy ordering matching the columns of `predicted_logits`.

    Returns accuracy, macro-F1, and mean held-out log-likelihood. A taxonomy that cannot beat
    the marginal-frequency baseline (also reported) is evidence the granularity is wrong.
    """
    idx = move_index(moves)
    y = np.array([idx[m] for m in moves_observed], dtype=int)
    logits = np.asarray(predicted_logits, dtype=float)
    if logits.shape != (len(y), len(moves)):
        raise ValueError(f"predicted_logits must be (n, |M|)={(len(y), len(moves))}")

    logp = logits - _logsumexp(logits, axis=1, keepdims=True)
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    ll = float(logp[np.arange(len(y)), y].mean())

    # macro-F1 over moves
    f1s = []
    for k in range(len(moves)):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)

    # marginal-frequency baseline log-likelihood
    counts = np.bincount(y, minlength=len(moves)).astype(float)
    base_p = np.clip(counts / counts.sum(), 1e-12, 1.0)
    base_ll = float(np.log(base_p[y]).mean())

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "mean_loglik": ll,
        "baseline_mean_loglik": base_ll,
        "loglik_gain_over_baseline": ll - base_ll,
    }


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
