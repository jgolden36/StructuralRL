"""Phase 1 `tier3` track — clean real-ish corpus: Stages 0-3 + merit test.

Runs the SAME Stages 0-3 + merit test as the sim track, but on **Tier 3** data: a version-
controlled computational-discovery repository (the cleanest real corpus, no wet lab). This
exercises the real data path (encoder, taxonomy induction, messy logs) where truth is unknown,
so there is no recovery-vs-truth check — only the pre-registered merit test and go/no-go.

SCAFFOLD: wired to `data.tier3_vcs.load_tier3_vcs` (not yet implemented) and the learned encoder.
The estimation/evaluation core is shared with `phase1_sim`; what differs is the front end (real
phi, induced taxonomy validation) and the absence of ground truth. Gate: pass go/no-go on Tier 3
before any real lab data (CLAUDE.md §9 step 7).
"""
from __future__ import annotations

from pathlib import Path

from ..data.tier3_vcs import load_tier3_vcs
from ._config import RunConfig


def run(config: RunConfig, repo_path: str | Path, out_dir: str | Path | None = None) -> dict:
    """Run Stages 0-3 + merit test on a Tier 3 corpus. Currently raises until the loader exists."""
    panel = load_tier3_vcs(repo_path, config={"corpus": "tier3"})  # NotImplementedError for now
    # Once the loader returns a real EventPanel:
    #   1. validate the induced move taxonomy (representation.moves.validate_taxonomy);
    #   2. fit phi (representation.encoder.build_torch_encoder) WITH phase/budget;
    #   3. Stage 1 LogitCCP + transition (or surrogate) ; Stage 2 second step (+ measured anchors,
    #      anchor.py, since Tier 3 logs compute cost / info gain);
    #   4. Stage 3 warm start; merit test + sufficiency + transfer + go/no-go.
    raise NotImplementedError("Tier 3 pipeline awaits data.tier3_vcs.load_tier3_vcs (build step 6).")
