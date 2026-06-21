"""Run-config loading + snapshotting (determinism discipline, CLAUDE.md §8).

Every pipeline run takes a seed and writes a snapshot recording normalization, lambda, beta,
tolerances, and corpus/tier, so recovery and merit-test runs are reproducible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    """Snapshot of everything that makes a run reproducible."""

    seed: int = 0
    corpus: str = "sim"  # sim | tier3 | tier1 | tier2
    beta: float = 0.9  # calibrated discount (sensitivity reported separately)
    lam: float | None = None  # pooling strength; None => select by CV
    lambdas: list[float] = field(default_factory=lambda: [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    normalization: str = "center_within_state"
    # synthetic-game knobs (sim track)
    n_states: int = 6
    n_moves: int = 4
    n_features: int = 3
    n_roles: int = 3
    n_campaigns: int = 120
    horizon: int = 12
    delta_scale: float = 0.4
    # merit-test tolerances (pre-registered)
    tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "role_move_freq": 0.10,
            "move_transition": 0.12,
            "role_transition": 0.12,
        }
    )
    sufficiency_alpha: float = 0.05
    transfer_tol: float = 0.05

    @classmethod
    def load(cls, path: str | Path | None) -> "RunConfig":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def snapshot(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "config_snapshot.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=_default))
        return path


def _default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")
