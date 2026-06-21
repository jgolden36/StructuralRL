"""StructuralRL — structural initialization for autonomous-laboratory discovery.

Notation follows the two working notes in the repository root (the source of truth):
    phi      state encoder  -> sufficient-statistic vector
    sigma_i  role-conditional CCPs  sigma_i(a | s)
    F        transition law / surrogate  F(s' | s, a)
    theta_i = theta_0 + delta_i   partially pooled flow payoffs
    beta     calibrated discount factor
    lambda   pooling strength;  lambda_KL  Stage-4 anchor strength
"""

__version__ = "0.0.1"
