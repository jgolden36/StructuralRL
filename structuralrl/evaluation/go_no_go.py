"""The pre-registered go / no-go decision rule (CLAUDE.md §6).

State the decision BEFORE running. The approach has merit (and Stage 4 / Phase 2 is worth
running) **iff all three hold**:

1. Stage 3 collective matches the pre-registered interaction statistics within stated tolerances;
2. the state-sufficiency falsification test does NOT reject;
3. the pooled objective theta_0 transfers to held-out campaigns.

Branches are encoded as explicit return states (an enum), NOT free text:

* statistics match but sufficiency rejects hard -> constraint is INSTRUMENTATION (richer logging
  of rationale/hypothesis before any improvement claim).
* theta_0 does not transfer -> no common discovery objective across sampled labs; retreat to
  HOMOGENEOUS STRATA (or finite mixtures).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GoNoGoState(str, Enum):
    GO = "go"  # all three hold; Stage 4 / Phase 2 authorized
    NO_GO_STATS = "no_go_statistics"  # interaction statistics do not match
    NO_GO_INSTRUMENTATION = "no_go_instrumentation"  # stats ok but sufficiency rejects hard
    NO_GO_STRATA = "no_go_strata"  # theta_0 does not transfer; retreat to homogeneous strata
    NO_GO_MULTIPLE = "no_go_multiple"  # more than one failure


@dataclass
class GoNoGoDecision:
    state: GoNoGoState
    stats_pass: bool
    sufficiency_pass: bool  # True == does NOT reject sufficiency (good)
    transfer_pass: bool
    summary: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def go(self) -> bool:
        return self.state is GoNoGoState.GO


def decide(
    stat_comparisons,
    sufficiency_result,
    transfer_result,
) -> GoNoGoDecision:
    """Apply the three-part pre-registered rule and return an explicit decision state.

    Parameters
    ----------
    stat_comparisons : list of interaction_stats.StatComparison (only `available` ones gate).
    sufficiency_result : evaluation.sufficiency_test.SufficiencyResult.
    transfer_result : evaluation.transfer_test.TransferResult.
    """
    available = [c for c in stat_comparisons if getattr(c, "available", True)]
    stats_pass = bool(available) and all(c.passed for c in available)
    sufficiency_pass = not sufficiency_result.rejects_sufficiency  # not rejecting is good
    transfer_pass = transfer_result.transfers

    failures = []
    if not stats_pass:
        failures.append(GoNoGoState.NO_GO_STATS)
    if not transfer_pass:
        failures.append(GoNoGoState.NO_GO_STRATA)
    # sufficiency only gates as "instrumentation" branch when stats otherwise pass
    if stats_pass and not sufficiency_pass:
        failures.append(GoNoGoState.NO_GO_INSTRUMENTATION)
    if not stats_pass and not sufficiency_pass:
        failures.append(GoNoGoState.NO_GO_INSTRUMENTATION)

    if not failures and sufficiency_pass:
        state = GoNoGoState.GO
    elif len(set(failures)) > 1:
        state = GoNoGoState.NO_GO_MULTIPLE
    else:
        state = failures[0]

    summary = (
        f"stats_pass={stats_pass}, sufficiency_pass={sufficiency_pass}, "
        f"transfer_pass={transfer_pass} -> {state.value}"
    )
    return GoNoGoDecision(
        state=state,
        stats_pass=stats_pass,
        sufficiency_pass=sufficiency_pass,
        transfer_pass=transfer_pass,
        summary=summary,
        detail={
            "interaction_stats": [c.__dict__ for c in stat_comparisons],
            "sufficiency": sufficiency_result.describe(),
            "transfer": transfer_result.describe(),
        },
    )
