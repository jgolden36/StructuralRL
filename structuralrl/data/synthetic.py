"""Phase 1 `sim` track — generate trajectories from a KNOWN dynamic game.

This is the only place synthetic data exists, and it is always labelled `provenance="synthetic"`
(CLAUDE.md §8 "No fabricated data" — synthetic *inputs* for the recovery unit test are allowed
and labelled; fabricated *results* are never allowed).

We use a small tabular **logit dynamic discrete choice** game with payoffs linear in known
features `x(s, a)`:

    pi_i(s, a) = x(s, a) . theta_i,     theta_i = theta_0 + delta_i   (partial pooling, Eq. pool)

Each role i plays its own quantal-response best policy on a shared transition law F(s'|s,a)
(the artifact evolves mechanically the same way; roles differ only in payoff weights). Because
truth (theta_0, {delta_i}, F) is known, Stages 0-2 can be checked against it — the canonical
correctness test for the estimator (cannot be done on real data where truth is unknown).

The quantal-response equilibrium is the standard soft/logit DP fixed point
(McKelvey-Palfrey smoothed best response), solved by value iteration:

    v_i(s, a) = x(s, a).theta_i + beta * sum_s' F(s'|s,a) V_i(s')
    V_i(s)    = logsumexp_a v_i(s, a)            (Type-I EV shocks, scale 1)
    sigma_i(a|s) = softmax_a v_i(s, a)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..representation.encoder import TabularEncoder
from ..representation.moves import MOVES, ROLES
from .schema import EventPanel, EventRecord, build_panel


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class TabularGame:
    """A known-truth tabular logit dynamic game. All arrays are ground truth.

    Attributes
    ----------
    P : (S, A, S) transition tensor F(s'|s, a) (row-stochastic over s').
    X : (S, A, k) structural payoff features x(s, a).
    beta : discount factor (calibrated, not estimated — see CLAUDE.md §7).
    theta0 : (k,) pooled objective.
    deltas : dict role -> (k,) role-specific deviation (shrunk toward 0).
    """

    P: np.ndarray
    X: np.ndarray
    beta: float
    theta0: np.ndarray
    deltas: dict[str, np.ndarray]
    roles: tuple[str, ...]
    moves: tuple[str, ...]  # the A move names this game uses (subset of MOVES)

    def __post_init__(self) -> None:
        self.S, self.A, self.k = self.X.shape
        assert self.P.shape == (self.S, self.A, self.S)
        assert set(self.deltas) == set(self.roles)

    def theta(self, role: str) -> np.ndarray:
        return self.theta0 + self.deltas[role]

    # --- equilibrium -------------------------------------------------------
    def solve_qre(self, tol: float = 1e-12, max_iter: int = 5000) -> dict[str, dict]:
        """Solve the logit QRE per role by value iteration (a contraction for beta < 1)."""
        out: dict[str, dict] = {}
        for role in self.roles:
            theta = self.theta(role)
            r = self.X @ theta  # (S, A) flow payoff
            V = np.zeros(self.S)
            for _ in range(max_iter):
                v = r + self.beta * (self.P @ V)  # (S, A)
                V_new = _logsumexp(v, axis=1)
                if np.max(np.abs(V_new - V)) < tol:
                    V = V_new
                    break
                V = V_new
            v = r + self.beta * (self.P @ V)
            out[role] = {"V": V, "v": v, "sigma": _softmax(v, axis=1)}
        return out

    # --- sampling ----------------------------------------------------------
    def sample(
        self,
        n_campaigns: int,
        horizon: int,
        seed: int = 0,
        budget0: float = 1.0,
        encoder: TabularEncoder | None = None,
    ) -> tuple[EventPanel, dict]:
        """Roll out campaigns under the QRE and emit an :class:`EventPanel` + transition tuples.

        Each campaign is an episode of length `horizon`; every role takes a turn each step
        (turn-taking DMTA structure). phase = t / horizon and a decremented budget enter phi(s),
        so the within-episode non-stationarity is carried in the state (state-sufficiency).

        Returns (panel, extras) where extras["transitions"] is a list of (s, a, s') for F-hat and
        extras["sa"] maps role -> (states, actions) integer arrays for the tabular CCP/second-step.
        """
        rng = np.random.default_rng(seed)
        eq = self.solve_qre()
        enc = encoder or TabularEncoder(self.S, include_phase_budget=True)

        records: list[EventRecord] = []
        transitions: list[tuple[int, int, int]] = []
        sa: dict[str, dict[str, list[int]]] = {r: {"s": [], "a": []} for r in self.roles}

        for c in range(n_campaigns):
            cid = f"sim-{c:04d}"
            s = int(rng.integers(self.S))
            budget = budget0
            for t in range(horizon):
                phase = t / max(horizon - 1, 1)
                # decrement budget by a per-step draw so it varies across campaigns
                budget = max(0.0, budget - rng.uniform(0.0, 1.5 / horizon))
                phi = enc.encode({"index": s, "phase": phase, "budget_remaining": budget})
                # one move per role this step, on the shared state s
                next_s_for_chain = None
                for role in self.roles:
                    a = int(rng.choice(self.A, p=eq[role]["sigma"][s]))
                    s_next = int(rng.choice(self.S, p=self.P[s, a]))
                    records.append(
                        EventRecord(
                            campaign_id=cid,
                            t=t,
                            role=role,
                            move=self.moves[a],
                            state_features=phi,
                            payload={"state_index": s, "action_index": a, "phase": phase},
                            costs={},  # no measured anchors in the pure-sim game
                        )
                    )
                    transitions.append((s, a, s_next))
                    sa[role]["s"].append(s)
                    sa[role]["a"].append(a)
                    if role == self.roles[-1]:
                        next_s_for_chain = s_next
                s = next_s_for_chain if next_s_for_chain is not None else s

        panel = build_panel(records, self.roles, self.moves, provenance="synthetic")
        extras = {
            "transitions": transitions,
            "sa": {r: (np.array(d["s"]), np.array(d["a"])) for r, d in sa.items()},
            "equilibrium": eq,
            "encoder": enc,
        }
        return panel, extras


def make_random_game(
    n_states: int = 6,
    n_moves: int = 4,
    n_features: int = 3,
    n_roles: int = 3,
    beta: float = 0.9,
    delta_scale: float = 0.4,
    seed: int = 0,
) -> TabularGame:
    """Construct a random known-truth game for the recovery test.

    `delta_scale` controls how far role payoffs deviate from the pooled theta0 (the thing partial
    pooling is meant to shrink). Features are dense and low-dimensional so theta is point
    identified up to the (typically trivial) within-state shaping subspace.
    """
    rng = np.random.default_rng(seed)
    roles = ROLES[:n_roles]
    moves = MOVES[:n_moves]

    # row-stochastic transition with some persistence
    logits = rng.normal(size=(n_states, n_moves, n_states))
    P = _softmax(logits, axis=2)

    X = rng.normal(size=(n_states, n_moves, n_features))
    theta0 = rng.normal(scale=1.0, size=n_features)
    deltas = {r: rng.normal(scale=delta_scale, size=n_features) for r in roles}

    return TabularGame(P=P, X=X, beta=beta, theta0=theta0, deltas=deltas, roles=roles, moves=moves)
