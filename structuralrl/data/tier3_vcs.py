"""Phase 1 `tier3` — parse a version-controlled computational-discovery repository.

Tier 3 (companion §"Version-controlled computational discovery") is the **cleanest real corpus**:
a version-controlled computational-materials / in-silico-discovery repo (commits, runs, reviews of
an in-silico campaign). It is the discovery analog of the version-controlled coding/writing
testbed and exercises the real data path (encoder, taxonomy induction, messy logs) WITHOUT a wet
lab. It carries revealed preference at the cost of messier role labels.

This is a SCAFFOLD with a concrete mapping plan, not a finished parser. Filling it in is build-
order step 6 (CLAUDE.md §9). NO raw repository data is committed (data governance, §8); this
loader takes a path/config and emits an :class:`EventPanel`.

Mapping plan (commit/PR graph -> DMTA event records):
  commit / issue opened ............ propose
  experiment/config added or queued  design
  CI run / simulation executed ..... execute
  result parsed / notebook output .. characterize
  review approve / merge ........... accept
  review request-changes / revert .. reject
  re-run / confirmation run ........ replicate
  branch redirection / scope change  pivot
  label 'blocked' / 'wontfix' ...... defer / stop
  escalation to maintainer ......... escalate
Role labels come from author identity + bot/automation tags; phase/budget from commit timeline
position and CI compute budget (the phase/budget coordinates state-sufficiency requires).
"""
from __future__ import annotations

from pathlib import Path

from .schema import EventPanel


def load_tier3_vcs(repo_path: str | Path, config: dict | None = None) -> EventPanel:
    """Parse a version-controlled computational-discovery repo into an :class:`EventPanel`.

    Not yet implemented — see the module docstring for the commit/PR -> DMTA mapping plan. Raises
    NotImplementedError so a pipeline that reaches for Tier 3 fails loudly rather than silently
    fabricating data.
    """
    raise NotImplementedError(
        "Tier 3 VCS loader is a scaffold (build-order step 6). Implement the commit/PR->DMTA "
        "mapping documented in this module, emitting schema.EventRecord with phase/budget in phi."
    )
