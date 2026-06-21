"""Phase 2 Tier 1 — self-driving-lab decision logs (richest panel).

Tier 1 (companion §"Some roles are already automated"): timestamped, role-labeled, state-
conditioned API-call logs from a self-driving lab. The cleanest possible panel — the designer is
a Bayesian-optimization policy, the executor a robot, the analyst an ML interpreter.

**Estimand caveat (carried in this docstring on purpose):** from a fully autonomous platform's
logs, the estimator recovers the platform's *encoded objective and coordination protocol*, NOT a
scientist's revealed preference. Useful (interpretable, improvable reward for an existing
pipeline) but a different estimand than Tier 2. Choose the corpus with the estimand in mind.

SCAFFOLD — Phase 2 only, and only after Phase 1 passes go/no-go. NO raw data committed.
"""
from __future__ import annotations

from pathlib import Path

from .schema import EventPanel


def load_tier1_platform(log_path: str | Path, config: dict | None = None) -> EventPanel:
    """Parse self-driving-lab API-call logs into an :class:`EventPanel` (not yet implemented)."""
    raise NotImplementedError(
        "Tier 1 platform loader is a Phase-2 scaffold. Map role-labeled API calls to DMTA moves; "
        "remember the estimand is the platform's encoded objective, not revealed preference."
    )
