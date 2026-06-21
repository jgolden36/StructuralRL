"""Canonical correctness check: the two-step estimator recovers KNOWN payoffs.

This is the unit test the whole suite is gated on (CLAUDE.md §8): it must pass before any real-
data run is trusted. We test recovery of the identified object theta_i = theta_0 + delta_i (the
theta_0/delta split is identified only under the pooling penalty, so we compare per-role payoffs).
"""
import numpy as np

from structuralrl.data.synthetic import make_random_game
from structuralrl.estimation import forward_sim
from structuralrl.estimation.second_step import penalized_pseudo_likelihood


def _sample_oracle(game, sigma_by_role, n_per_role, seed=0):
    """Draw (s, a) for each role: states uniform over S, actions ~ sigma_true (full coverage)."""
    rng = np.random.default_rng(seed)
    data = {}
    for r in game.roles:
        s = rng.integers(game.S, size=n_per_role)
        a = np.array([rng.choice(game.A, p=sigma_by_role[r][si]) for si in s])
        data[r] = (s, a)
    return data


def test_recovers_true_payoffs_oracle():
    game = make_random_game(n_states=6, n_moves=4, n_features=3, n_roles=3, seed=7)
    eq = game.solve_qre()
    sigma_by_role = {r: eq[r]["sigma"] for r in game.roles}

    psi_by_role, c_by_role = {}, {}
    for r in game.roles:
        psi_by_role[r], c_by_role[r] = forward_sim.tabular_feature_sums(
            sigma_by_role[r], game.P, game.X, game.beta
        )

    data = _sample_oracle(game, sigma_by_role, n_per_role=40000, seed=7)
    res = penalized_pseudo_likelihood(data, psi_by_role, c_by_role, game.k, lam=0.0)
    assert res.success

    for r in game.roles:
        err = np.max(np.abs(res.theta(r) - game.theta(r)))
        assert err < 0.15, f"role {r}: max|theta-hat - theta| = {err:.3f}"


def test_pipeline_runs_end_to_end():
    from structuralrl.evaluation.go_no_go import GoNoGoState
    from structuralrl.pipelines._config import RunConfig
    from structuralrl.pipelines.phase1_sim import run

    cfg = RunConfig(seed=3, n_campaigns=120, horizon=12, delta_scale=0.2, lam=0.1)
    res = run(cfg)

    # the recovery report exists and is reasonable on finite data
    assert res["recovery"]["max_abs_err"] < 0.6
    # the go/no-go machinery returns a valid, explicit decision state
    assert res["decision"].state in set(GoNoGoState)
    # interaction-statistics families that ARE available were scored
    available = [c for c in res["interaction_stats"] if c.available]
    assert available and all(np.isfinite(c.distance) for c in available)
