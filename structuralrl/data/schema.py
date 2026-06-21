"""Canonical event record shared by every data tier (Stage 0 input contract).

Every loader (`synthetic`, `tier3_vcs`, `tier1_platform`, `tier2_human`) must emit a
sequence of :class:`EventRecord` / an :class:`EventPanel`, so that downstream stages
(encoder, CCP, transition, second step) are corpus-agnostic.

Schema (one row = one *move* by one role at one campaign timestep):

    campaign_id     str    episode / campaign identifier (an episode = a DMTA campaign)
    t               int    integer timestep within the campaign (0-indexed)
    role            str    functional role that moved (see roles.ROLES)
    move            str    discrete move type from the taxonomy (moves.MOVES)
    payload         dict   realized within-move structured choice (specific compound/protocol);
                           NOT used by the discrete structural step, kept for audit/Stage-4.
    state_features  ndarray  phi(s_t): encoded state, MUST include campaign phase / budget-remaining
                             (state-sufficiency assumption, companion note).
    costs           dict   externally-denominated, *measured* payoff anchors c(s,a) (anchor.py):
                           reagent_cost, instrument_cost, compute_cost, wall_clock, info_gain.

`labels` distinguishes learned / estimated / assumed provenance per the paper's discipline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --- canonical column names (single source of truth for loaders) -------------
COLUMNS = ("campaign_id", "t", "role", "move", "payload", "state_features", "costs")

# Measured payoff-anchor keys (companion Eq. anchor). Levels are *observed*, not free.
COST_KEYS = ("reagent_cost", "instrument_cost", "compute_cost", "wall_clock", "info_gain")


@dataclass(frozen=True)
class EventRecord:
    """A single role move at a single campaign timestep. See module docstring."""

    campaign_id: str
    t: int
    role: str
    move: str
    state_features: np.ndarray
    payload: Mapping[str, Any] = field(default_factory=dict)
    costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        feats = np.asarray(self.state_features, dtype=float)
        object.__setattr__(self, "state_features", feats)
        if feats.ndim != 1:
            raise ValueError(f"state_features must be a 1-D phi(s) vector, got shape {feats.shape}")
        if self.t < 0:
            raise ValueError("t must be a non-negative campaign timestep")


@dataclass
class EventPanel:
    """An ordered collection of :class:`EventRecord`, grouped by campaign.

    This is the Stage-0 hand-off object: a clean panel that every estimator consumes.
    """

    records: list[EventRecord]
    roles: tuple[str, ...]
    moves: tuple[str, ...]
    # Provenance discipline: 'learned' | 'estimated' | 'assumed' | 'measured' | 'synthetic'.
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        self.records.sort(key=lambda r: (r.campaign_id, r.t))

    # --- convenience views ---------------------------------------------------
    @property
    def n_campaigns(self) -> int:
        return len(set(r.campaign_id for r in self.records))

    @property
    def feature_dim(self) -> int:
        return 0 if not self.records else self.records[0].state_features.shape[0]

    def by_campaign(self) -> dict[str, list[EventRecord]]:
        out: dict[str, list[EventRecord]] = {}
        for r in self.records:
            out.setdefault(r.campaign_id, []).append(r)
        return out

    def by_role(self) -> dict[str, list[EventRecord]]:
        out: dict[str, list[EventRecord]] = {role: [] for role in self.roles}
        for r in self.records:
            out.setdefault(r.role, []).append(r)
        return out

    def to_frame(self) -> pd.DataFrame:
        """Flatten to a pandas frame for inspection / parquet I/O (NOT for raw-data commit)."""
        rows = []
        for r in self.records:
            row: dict[str, Any] = {
                "campaign_id": r.campaign_id,
                "t": r.t,
                "role": r.role,
                "move": r.move,
                "payload": dict(r.payload),
            }
            row.update({f"phi_{i}": v for i, v in enumerate(r.state_features)})
            row.update({k: r.costs.get(k, np.nan) for k in COST_KEYS})
            rows.append(row)
        return pd.DataFrame(rows)

    def validate(self) -> None:
        """Cheap structural checks; raises on a malformed panel."""
        dim = self.feature_dim
        for r in self.records:
            if r.role not in self.roles:
                raise ValueError(f"unknown role {r.role!r}; declared roles={self.roles}")
            if r.move not in self.moves:
                raise ValueError(f"unknown move {r.move!r}; declared moves={self.moves}")
            if r.state_features.shape[0] != dim:
                raise ValueError("ragged phi(s): all state_features must share a dimension")


def build_panel(
    records: Iterable[EventRecord],
    roles: Sequence[str],
    moves: Sequence[str],
    provenance: str = "unspecified",
) -> EventPanel:
    """Construct and validate an :class:`EventPanel` from loose records."""
    panel = EventPanel(list(records), tuple(roles), tuple(moves), provenance)
    panel.validate()
    return panel
