"""Pre-registered held-out interaction statistics (the merit test — companion note §5).

**Statistics are named in advance so success cannot be selected after the fact. Do NOT add
post-hoc metrics to a passing run** (CLAUDE.md §5). The five families:

1. Role-conditional move frequencies.
2. Move-transition matrix (which role acts after which / after which move — the DMTA cadence).
3. Inter-move & outcome timing (intervals, time-to-result, experiments-to-discovery).
4. State-conditional response profiles for the pre-registered event set.
5. Campaign-level outcomes incl. the **false-discovery / irreproducibility rate** — the single
   most important statistic, the tail quantity coordination most protects.

`compare` scores the Stage-3 collective's statistics against held-out human statistics within
stated tolerances and returns a structured, per-statistic pass/fail report (consumed by
go_no_go.py). Families 3-5 require outcome fields that the pure-`sim` track does not generate;
they are computed where present and reported as `unavailable` otherwise, never silently skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..representation.moves import MOVES, ROLES


def _normalize(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else counts


def role_move_frequencies(
    events,
    roles=ROLES,
    moves=MOVES,
) -> np.ndarray:
    """(|roles|, |moves|) matrix of P(move | role). `events` is an iterable of dicts/records with
    `.role`/`.move` or ['role']/['action'] keys (rollout log or EventPanel records)."""
    ri = {r: i for i, r in enumerate(roles)}
    mi = {m: i for i, m in enumerate(moves)}
    counts = np.zeros((len(roles), len(moves)))
    for e in events:
        role, move = _role_move(e, moves)
        if role in ri and move in mi:
            counts[ri[role], mi[move]] += 1
    return np.vstack([_normalize(counts[i]) for i in range(len(roles))])


def move_transition_matrix(events_by_campaign, moves=MOVES) -> np.ndarray:
    """(|moves|, |moves|) row-stochastic P(next move | current move), within campaigns.

    `events_by_campaign` maps campaign_id -> ordered list of events. Captures the realized
    design-make-test-analyze cadence at the move level.
    """
    mi = {m: i for i, m in enumerate(moves)}
    counts = np.zeros((len(moves), len(moves)))
    for seq in events_by_campaign.values():
        prev = None
        for e in seq:
            _, move = _role_move(e, moves)
            if move not in mi:
                prev = None
                continue
            if prev is not None:
                counts[mi[prev], mi[move]] += 1
            prev = move
    return np.vstack([_normalize(counts[i]) for i in range(len(moves))])


def role_transition_matrix(events_by_campaign, roles=ROLES) -> np.ndarray:
    """(|roles|, |roles|) row-stochastic P(next acting role | current acting role)."""
    ri = {r: i for i, r in enumerate(roles)}
    counts = np.zeros((len(roles), len(roles)))
    for seq in events_by_campaign.values():
        prev = None
        for e in seq:
            role, _ = _role_move(e, MOVES)
            if role not in ri:
                prev = None
                continue
            if prev is not None:
                counts[ri[prev], ri[role]] += 1
            prev = role
    return np.vstack([_normalize(counts[i]) for i in range(len(roles))])


@dataclass
class StatComparison:
    name: str
    distance: float
    tolerance: float
    passed: bool
    available: bool = True
    detail: dict = field(default_factory=dict)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Mean total-variation distance between matched rows of two stochastic matrices."""
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    return float(np.mean(0.5 * np.abs(p - q).sum(axis=1)))


def compare(
    collective_events,
    human_events,
    tolerances: dict | None = None,
    roles=ROLES,
    moves=MOVES,
) -> list[StatComparison]:
    """Compare collective vs human interaction statistics within pre-registered tolerances.

    `*_events` may be either a flat list (for frequency stats) or a {campaign: [events]} mapping
    (enables transition stats). Returns one :class:`StatComparison` per pre-registered family;
    families needing outcome data absent from `sim` are flagged `available=False`.
    """
    tol = {"role_move_freq": 0.10, "move_transition": 0.12, "role_transition": 0.12}
    tol.update(tolerances or {})

    coll_flat = _flatten(collective_events)
    hum_flat = _flatten(human_events)
    out = [
        _cmp(
            "role_move_freq",
            role_move_frequencies(coll_flat, roles, moves),
            role_move_frequencies(hum_flat, roles, moves),
            tol["role_move_freq"],
        )
    ]

    if isinstance(collective_events, dict) and isinstance(human_events, dict):
        out.append(
            _cmp(
                "move_transition",
                move_transition_matrix(collective_events, moves),
                move_transition_matrix(human_events, moves),
                tol["move_transition"],
            )
        )
        out.append(
            _cmp(
                "role_transition",
                role_transition_matrix(collective_events, roles),
                role_transition_matrix(human_events, roles),
                tol["role_transition"],
            )
        )

    # Families 3-5 need outcome fields (timing, validated-discovery flags, reproducibility).
    for name in ("inter_move_timing", "event_response_profiles", "campaign_outcomes_fdr"):
        out.append(StatComparison(name, np.nan, tol.get(name, np.nan), False, available=False))
    return out


def false_discovery_rate(campaign_outcomes) -> float:
    """The single most important statistic: validated-as-true that fail replication.

    `campaign_outcomes` is an iterable of dicts with bool keys `claimed_discovery` and
    `replicated`. FDR = P(not replicated | claimed). Requires real/tier3 outcome data; raises if
    no claimed discoveries are present (do not report a spurious 0).
    """
    claimed = [o for o in campaign_outcomes if o.get("claimed_discovery")]
    if not claimed:
        raise ValueError("no claimed discoveries in outcomes; FDR is undefined on this corpus")
    failed = sum(1 for o in claimed if not o.get("replicated"))
    return failed / len(claimed)


# --- helpers ----------------------------------------------------------------
def _role_move(e, moves):
    if hasattr(e, "role"):
        role = e.role
        move = getattr(e, "move", None)
    else:
        role = e.get("role")
        move = e.get("move")
        if move is None and "action" in e:  # rollout logs store integer action ids
            ai = e["action"]
            move = moves[ai] if isinstance(ai, (int, np.integer)) and ai < len(moves) else ai
    return role, move


def _flatten(events):
    if isinstance(events, dict):
        return [e for seq in events.values() for e in seq]
    return list(events)


def _cmp(name, p, q, tol):
    d = _tvd(p, q)
    return StatComparison(name, d, tol, d <= tol, available=True)
