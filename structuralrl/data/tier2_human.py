"""Phase 2 Tier 2 — human-led campaign records (ELN / LIMS / instrument logs).

Tier 2 (companion §"Some roles are already automated"): records from an instrumented human-led
high-throughput campaign. **Carries revealed preference** (the desirable estimand), at the cost
of messier role labels and more unlogged (tacit) judgment — which is exactly what the state-
sufficiency falsification test is designed to detect.

SCAFFOLD — Phase 2 only, and only after Phase 1 passes go/no-go. NO raw data committed.
Heterogeneity across labs is large: never naively pool labs that played different equilibria
(use finite mixtures or homogeneous strata; CLAUDE.md §5/§7).
"""
from __future__ import annotations

from pathlib import Path

from .schema import EventPanel


def load_tier2_human(records_path: str | Path, config: dict | None = None) -> EventPanel:
    """Parse ELN/LIMS/instrument logs into an :class:`EventPanel` (not yet implemented)."""
    raise NotImplementedError(
        "Tier 2 human-led loader is a Phase-2 scaffold. Map ELN/LIMS/instrument events to DMTA "
        "moves; expect messy role labels and tacit judgment (run the sufficiency test)."
    )
