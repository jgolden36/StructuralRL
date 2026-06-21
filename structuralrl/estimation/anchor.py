"""Stage 2 — anchored payoff decomposition pi = c + x^T theta (companion Eq. anchor).

The discovery domain is favorable on normalization because several payoff components are
**measured and externally denominated**. Write

    pi_i(s, a_i, a_-i; theta_i) = c(s, a_i, a_-i) + x(s, a_i, a_-i)^T theta_i,   (Eq. anchor)

where c collects components whose LEVELS are observed:
  - negative realized reagent + instrument cost,
  - negative compute cost + wall-clock,
  - realized surrogate information gain (posterior entropy / variance reduction).

Because the levels in c are not free, the anchor shrinks the shaping-equivalence class — the
**strongest** of the three normalization disciplines (sensitivity < invariance < anchoring), and
available here by construction of how discovery campaigns are instrumented (CLAUDE.md §4).

Operationally: the anchor enters the choice-specific value as a KNOWN, theta-independent term.
Its discounted forward-simulated sum is added to the logit offset c(s,a) from forward_sim, so the
second step estimates only the FREE part x^T theta. This module builds that anchor offset from
the measured `costs` carried in the event schema.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import COST_KEYS

# Sign convention turning measured costs into payoff *levels* (Eq. anchor):
# costs are negative payoff; information gain is positive payoff.
ANCHOR_SIGNS = {
    "reagent_cost": -1.0,
    "instrument_cost": -1.0,
    "compute_cost": -1.0,
    "wall_clock": -1.0,
    "info_gain": +1.0,
}


def measured_anchor_level(costs: dict) -> float:
    """Map a record's measured `costs` to the anchored payoff level c(s, a) (a scalar)."""
    return float(sum(ANCHOR_SIGNS[k] * float(costs.get(k, 0.0)) for k in COST_KEYS))


def anchor_table(
    costs_by_sa: dict[tuple[int, int], dict],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """Build the measured anchor c_anchor(s, a) table from per-(s,a) measured costs.

    Missing (s, a) entries are 0 (no measured anchor). The result is added to the forward-sim
    discounted anchor stream before the second step, so that x^T theta is the only free part.
    """
    c = np.zeros((n_states, n_actions))
    for (s, a), costs in costs_by_sa.items():
        c[int(s), int(a)] = measured_anchor_level(costs)
    return c


def discounted_anchor_offset(
    c_anchor: np.ndarray,
    sigma: np.ndarray,
    P: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Discounted forward-simulated sum of the measured anchor, as a logit offset (S, A).

    Same forward-simulation structure as forward_sim, applied to the KNOWN anchor flow instead of
    the entropy stream: the anchor contributes a theta-independent term to every choice-specific
    value, which we add to the logit offset so the second step recovers only the free payoff.
    """
    S, A = c_anchor.shape
    T = np.einsum("sa,sat->st", sigma, P)
    Minv = np.linalg.inv(np.eye(S) - beta * T)
    cbar = np.einsum("sa,sa->s", sigma, c_anchor)  # (S,)
    V_anchor = Minv @ cbar  # (S,)
    return c_anchor + beta * np.einsum("sat,t->sa", P, V_anchor)
