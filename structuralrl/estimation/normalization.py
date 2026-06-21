"""Shaping-equivalence-class bookkeeping (Hotz-Miller invariants).

Identification discipline (parent §"Normalization", CLAUDE.md §7):

* Flow payoffs are identified only up to the **shaping-equivalence class**: the Hotz-Miller
  inversion recovers *differences* in choice-specific values, the shaping-invariant object.
  Adding any state potential `Phi(s')` (potential-based shaping) or a per-state constant to the
  logits leaves observed policies unchanged.
* Payoff **scale** and **logit scale** are not separately identified; the shock scale is
  normalized to one, so recovered payoffs are in shock units (the LLM temperature inherits this).
* **Stage 3 is invariant** to the chosen normalization; **Stage 4 joint improvement is not**.

This module does not *resolve* the non-identification — it *records* the convention so a run is
reproducible and so Stage 4 can report sensitivity across admissible normalizations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

NormKind = Literal["none", "reference_action", "center_within_state", "anchored"]


@dataclass(frozen=True)
class Normalization:
    """A persisted normalization choice (written into the run config snapshot).

    kind:
        "none"               -- no level convention imposed (report sets, not points).
        "reference_action"   -- payoff of `ref_action` set to 0 in every state (Magnac-Thesmar).
        "center_within_state"-- logits demeaned within state (a representative of the class).
        "anchored"           -- part of the payoff carries measured levels (companion Eq. anchor);
                                the strongest discipline, shrinks the equivalence class.
    """

    kind: NormKind = "center_within_state"
    ref_action: int | None = None
    # Scale convention: shock scale fixed to 1 (recovered payoffs in shock units).
    shock_scale: float = 1.0
    notes: str = ""
    anchored_components: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return f"Normalization(kind={self.kind}, ref_action={self.ref_action}, shock_scale={self.shock_scale})"


def center_within_state(logits: np.ndarray) -> np.ndarray:
    """Subtract the per-state mean from logits (a representative of the shaping class).

    `logits` is (S, A). Softmax-invariant, used to put recovered and true value functions in a
    common gauge before comparison.
    """
    logits = np.asarray(logits, dtype=float)
    return logits - logits.mean(axis=-1, keepdims=True)


def reference_action(logits: np.ndarray, ref: int) -> np.ndarray:
    """Subtract the reference action's value in every state (Magnac-Thesmar convention)."""
    logits = np.asarray(logits, dtype=float)
    return logits - logits[..., [ref]]


def gauge_payoff_features(psi: np.ndarray) -> np.ndarray:
    """Demean forward-simulated feature sums within state.

    Only within-state differences of `psi @ theta` enter the logit, so `theta` is identified
    only on the row space of the within-state-demeaned `psi`. Returning the demeaned `psi`
    exposes exactly the identified directions; the orthogonal complement is the un-identified
    (shaping) subspace. `psi` is (S, A, k); returns the same shape, demeaned over A.
    """
    psi = np.asarray(psi, dtype=float)
    return psi - psi.mean(axis=1, keepdims=True)
