"""The Hotz-Miller invariant that makes recovery possible.

If the forward-simulated feature sums psi and entropy offset c are correct, then at the TRUE
payoff the implied logit policy softmax(psi @ theta + c) reproduces the data-generating policy
sigma_true exactly. This is the precondition for the second step to recover theta, so we test it
directly and tightly.
"""
import numpy as np

from structuralrl.data.synthetic import make_random_game
from structuralrl.estimation import forward_sim


def test_implied_policy_at_true_theta_matches_qre():
    game = make_random_game(n_states=6, n_moves=4, n_features=3, n_roles=3, seed=1)
    eq = game.solve_qre()
    for role in game.roles:
        sigma = eq[role]["sigma"]
        psi, c = forward_sim.tabular_feature_sums(sigma, game.P, game.X, game.beta)
        logits = psi @ game.theta(role) + c
        logits -= logits.max(axis=1, keepdims=True)
        implied = np.exp(logits)
        implied /= implied.sum(axis=1, keepdims=True)
        assert np.max(np.abs(implied - sigma)) < 1e-8


def test_feature_sums_are_finite_and_shaped():
    game = make_random_game(seed=2)
    eq = game.solve_qre()
    sigma = eq[game.roles[0]]["sigma"]
    psi, c = forward_sim.tabular_feature_sums(sigma, game.P, game.X, game.beta)
    assert psi.shape == (game.S, game.A, game.k)
    assert c.shape == (game.S, game.A)
    assert np.all(np.isfinite(psi)) and np.all(np.isfinite(c))
