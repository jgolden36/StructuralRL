"""Phase 2 — real lab data: Stages 0-3 baseline + Stage 4 improvement.

ONLY attempted if Phase 1 passes go/no-go (CLAUDE.md §2). Choose the corpus with the ESTIMAND in
mind: Tier 1 (autonomous-platform logs) recovers the platform's encoded objective; Tier 2
(human-led records) carries revealed preference.

Phase 2 runs Stages 0-3 to establish the baseline, then — and only then — Stage 4 improvement
in silico against a validated surrogate F-hat, under a KL/behavioral anchor to the Stage-3
policies that preserves coordination structure and false-discovery discipline. The sim-to-real
gap bounds how far improvement may travel from the explored region (transition.SurrogateTransition
support check).

SCAFFOLD: wired to the Tier 1/2 loaders and `agents.improve` (both intentionally unimplemented).
"""
from __future__ import annotations

from pathlib import Path

from ..agents.improve import ImproveConfig, improve
from ..data.tier1_platform import load_tier1_platform
from ..data.tier2_human import load_tier2_human
from ._config import RunConfig


def run(
    config: RunConfig,
    data_path: str | Path,
    phase1_decision,
    surrogate,
    out_dir: str | Path | None = None,
) -> dict:
    """Stages 0-3 baseline + (guarded) Stage 4 improvement on real lab data.

    `phase1_decision` is the Phase-1 GoNoGoDecision; Stage 4 refuses to run unless it is a GO.
    `surrogate` is a transition.SurrogateTransition supplying validated F-hat + support checks.
    """
    if config.corpus == "tier1":
        _ = load_tier1_platform(data_path, config={"corpus": "tier1"})  # NotImplementedError
    elif config.corpus == "tier2":
        _ = load_tier2_human(data_path, config={"corpus": "tier2"})  # NotImplementedError
    else:
        raise ValueError("phase2_real requires corpus in {'tier1','tier2'}")

    # ... Stages 0-3 (shared with phase1) produce `collective` ...
    # collective = build_collective_from_real_data(...)
    # return improve(collective, surrogate, phase1_decision, ImproveConfig(beta=config.beta))
    raise NotImplementedError(
        "Phase 2 pipeline awaits Tier 1/2 loaders and the Stage-4 trainer (build step 8). "
        "Stage 4 is gated on a passing Phase-1 go/no-go decision."
    )
