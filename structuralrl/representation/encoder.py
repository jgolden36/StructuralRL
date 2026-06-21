"""Stage 0 — state encoder phi(s) -> sufficient-statistic vector.

The whole structural argument rests on **Assumption (state sufficiency)**: phi(s) is a
sufficient statistic for continuation play. The encoder therefore MUST surface campaign phase /
budget-remaining so the within-episode non-stationarity lives in the *state*, not silently in
the payoff (companion note §"Episodic and non-stationary structure"). `sufficiency_test.py`
falsifies this assumption empirically.

Two encoders are provided:

* :class:`TabularEncoder` — identity / one-hot over a finite discrete state space. Used by the
  Phase 1 `sim` track where the state is a known integer; keeps the estimator exactly testable.
* :class:`TorchStateEncoder` — a learned MLP/sequence encoder for real corpora (Tier 1/2/3).
  Imported lazily so the core estimator install needs no torch.

Both expose `encode(state) -> np.ndarray` returning phi(s), and guarantee the phase/budget
coordinates are present via `with_phase_budget`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class StateEncoder(Protocol):
    """Interface every phi must satisfy."""

    feature_dim: int

    def encode(self, state) -> np.ndarray:  # pragma: no cover - protocol
        ...


def with_phase_budget(features: np.ndarray, phase: float, budget_remaining: float) -> np.ndarray:
    """Append the (phase, budget_remaining) coordinates that state-sufficiency requires.

    `phase` in [0, 1] is fraction of the campaign horizon elapsed; `budget_remaining` in [0, 1].
    Centralized here so no encoder can forget them (the silent-misestimation failure mode).
    """
    feats = np.asarray(features, dtype=float).ravel()
    return np.concatenate([feats, [float(phase), float(budget_remaining)]])


@dataclass
class TabularEncoder:
    """One-hot encoder over a finite discrete state space (Phase 1 `sim`).

    `include_phase_budget` keeps the interface honest even in the tabular case: the synthetic
    generator passes phase/budget so the same sufficiency machinery applies.
    """

    n_states: int
    include_phase_budget: bool = True

    @property
    def feature_dim(self) -> int:
        return self.n_states + (2 if self.include_phase_budget else 0)

    def encode(self, state) -> np.ndarray:
        s = int(state["index"] if isinstance(state, dict) else state)
        onehot = np.zeros(self.n_states, dtype=float)
        onehot[s] = 1.0
        if not self.include_phase_budget:
            return onehot
        if isinstance(state, dict):
            phase = float(state.get("phase", 0.0))
            budget = float(state.get("budget_remaining", 1.0))
        else:
            phase, budget = 0.0, 1.0
        return with_phase_budget(onehot, phase, budget)


def build_torch_encoder(*args, **kwargs):
    """Lazy factory for the learned encoder; requires the `nn` extra (torch).

    Kept behind a factory so importing `representation.encoder` never imports torch.
    """
    from ._torch_encoder import TorchStateEncoder  # noqa: WPS433 (lazy import by design)

    return TorchStateEncoder(*args, **kwargs)
