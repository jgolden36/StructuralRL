"""Stage-0 contract tests: event schema validation and taxonomy validation machinery."""
import numpy as np
import pytest

from structuralrl.data.schema import EventRecord, build_panel
from structuralrl.representation.moves import MOVES, ROLES, validate_taxonomy


def test_event_record_requires_1d_phi():
    with pytest.raises(ValueError):
        EventRecord("c", 0, "proposer", "propose", state_features=np.zeros((2, 2)))


def test_panel_validates_roles_and_moves():
    rec = EventRecord("c", 0, "proposer", "propose", state_features=np.zeros(3))
    panel = build_panel([rec], ROLES, MOVES, provenance="synthetic")
    assert panel.n_campaigns == 1
    assert panel.feature_dim == 3
    bad = EventRecord("c", 1, "proposer", "not_a_move", state_features=np.zeros(3))
    with pytest.raises(ValueError):
        build_panel([bad], ROLES, MOVES)


def test_taxonomy_validation_beats_baseline_on_separable_data():
    # construct logits that perfectly predict the held-out move -> positive gain over baseline
    n, m = 50, len(MOVES)
    y = np.arange(n) % m
    logits = np.full((n, m), -5.0)
    logits[np.arange(n), y] = 5.0
    report = validate_taxonomy([MOVES[i] for i in y], logits)
    assert report["accuracy"] == 1.0
    assert report["loglik_gain_over_baseline"] > 0
