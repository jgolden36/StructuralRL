"""Learned state encoder phi(s) for real corpora (Tier 1/2/3). Requires the `nn` extra.

This is a SCAFFOLD: it fixes the interface (a sequence/MLP encoder producing phi(s) with the
phase/budget coordinates concatenated) so real-data loaders can target it, but it is not yet
trained or validated. State-sufficiency (Assumption) must be checked with
`evaluation/sufficiency_test.py` before any recovered payoff on real data is trusted.
"""
from __future__ import annotations

import numpy as np

try:  # torch is an optional extra
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "TorchStateEncoder requires the 'nn' extra: pip install -e '.[nn]'"
    ) from exc


class TorchStateEncoder(nn.Module):
    """MLP encoder over a precomputed raw-feature vector, with phase/budget appended.

    Replace/extend with a sequence model (transformer over the campaign log) for real data.
    The forward pass deliberately keeps the last two input coordinates (phase, budget) routed
    straight through so they remain explicit in phi(s).
    """

    def __init__(self, raw_dim: int, hidden: int = 128, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim + 2  # + (phase, budget) passthrough
        self.net = nn.Sequential(
            nn.Linear(raw_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    @property
    def feature_dim(self) -> int:
        return self.out_dim

    def forward(self, raw: "torch.Tensor", phase_budget: "torch.Tensor") -> "torch.Tensor":
        h = self.net(raw)
        return torch.cat([h, phase_budget], dim=-1)

    def encode(self, state) -> np.ndarray:
        raw = torch.as_tensor(np.asarray(state["raw"], dtype="float32"))
        pb = torch.as_tensor(
            np.asarray([state.get("phase", 0.0), state.get("budget_remaining", 1.0)], "float32")
        )
        with torch.no_grad():
            return self.forward(raw.unsqueeze(0), pb.unsqueeze(0)).squeeze(0).cpu().numpy()
