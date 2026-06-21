"""Stage 4 — improvement (Phase 2 ONLY; do not run before Phase 1 passes go/no-go).

Multi-agent RL against F-hat on the recovered reward, minus an anchor to the Stage-3 policies:

    maximize  E[ sum_t beta^t ( theta_i . x(s,a) ) ]  -  lambda_KL * KL( pi_i || sigma_i-hat ).

The KL anchor preserves coordination structure and the false-discovery discipline (the domain's
safety-like property); it is the structural cousin of offline-RL pessimism (companion note).

This module is a **deliberate SCAFFOLD with guardrails**, not a finished trainer — Stage 4 is
out of scope for Phase 1, and the notes are emphatic about the conditions under which it is even
well-posed:

* **Normalization caveat.** Improving ONE role against frozen rivals is invariant across the
  shaping-equivalence class (a single-agent problem, potential-shaping-protected). JOINT movement
  of all roles is NOT invariant — discipline joint-improvement claims with the Eq.-anchor levels,
  sensitivity across normalizations, and normalization-invariant (externally measured) evaluation.
* **Off-path risk.** Off-path beliefs are not identified from on-path data, and the surrogate
  F-hat is reliable only near the explored region. Rollouts leaving the surrogate's validated
  support must be flagged (transition.SurrogateTransition.off_support_fraction) and bounded.

The class therefore refuses to run unless a go/no-go pass is asserted and a surrogate with a
support check is supplied. The actual policy-gradient/MARL loop is left as a typed TODO.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.go_no_go import GoNoGoDecision
from .collective import Collective


@dataclass
class ImproveConfig:
    lambda_kl: float = 1.0  # anchor strength to sigma_i-hat (preserves coordination / FDR)
    beta: float = 0.9
    joint: bool = False  # joint movement is NOT normalization-invariant (see module docstring)
    max_off_support_fraction: float = 0.05  # abort if rollouts leave the surrogate's support


class Stage4NotAuthorized(RuntimeError):
    """Raised when Stage 4 is invoked without a passing Phase-1 go/no-go decision."""


def improve(
    collective: Collective,
    surrogate,
    decision: GoNoGoDecision,
    config: ImproveConfig | None = None,
):
    """Entry point for Stage-4 improvement. Guardrailed; the MARL loop is a TODO (Phase 2).

    Parameters
    ----------
    collective : the Stage-3 warm-started collective (anchor target sigma_i-hat lives here).
    surrogate  : transition.SurrogateTransition (validated F-hat + support check).
    decision   : the pre-registered go/no-go outcome; Stage 4 only runs on a GO.
    """
    config = config or ImproveConfig()
    if not decision.go:
        raise Stage4NotAuthorized(
            "Stage 4 requires a passing Phase-1 go/no-go decision; got "
            f"go={decision.go} (reason: {decision.summary})."
        )
    if config.joint:
        # Allowed, but the caller must own the normalization discipline; we record the warning.
        collective.meta["stage4_joint_warning"] = (
            "Joint improvement is normalization-DEPENDENT; report sensitivity across "
            "normalizations and evaluate on externally measured outcomes only."
        )

    raise NotImplementedError(
        "Stage 4 MARL loop is a Phase-2 deliverable and intentionally unimplemented. "
        "Implement KL-anchored multi-agent policy gradient on the recovered reward against the "
        "surrogate here, aborting when surrogate.off_support_fraction exceeds "
        f"{config.max_off_support_fraction}."
    )
