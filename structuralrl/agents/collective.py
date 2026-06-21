"""Stage 3 — equilibrium-consistent initialization of the agent collective.

For each role:  policy <- sigma_i-hat,  reward <- theta_0-hat + delta_i-hat.

The collective now sits at an *estimate of the human discovery equilibrium* — the orchestration
baseline and auditable default, and the starting point Stage 4 improves from. **This stage is
fully normalization-invariant**: reproducing the estimated policies needs no payoff levels at all
(parent §"Normalization"), so nothing here depends on the chosen gauge.

The collective is deliberately lightweight: it carries each role's CCP (for acting) and recovered
reward (for Stage 4 and for audit), plus the F-hat / surrogate it was estimated against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..estimation.second_step import SecondStepResult


@dataclass
class RoleAgent:
    """A single warm-started role: acts by its recovered CCP, carries its recovered reward."""

    role: str
    sigma: np.ndarray  # (S, A) sigma_i-hat  (the policy; Stage-3 uses ONLY this)
    theta: np.ndarray  # (k,) theta_0-hat + delta_i-hat  (the reward; for Stage 4 / audit)
    psi: np.ndarray | None = None  # (S, A, k) feature sums, if Stage 4 will reuse them
    c: np.ndarray | None = None  # (S, A) logit offset

    def act(self, state: int, rng: np.random.Generator) -> int:
        """Sample a move from the recovered CCP at `state` (normalization-invariant)."""
        return int(rng.choice(self.sigma.shape[1], p=self.sigma[int(state)]))

    def reward(self, psi_sa: np.ndarray) -> float:
        """Recovered flow reward for a forward-sim feature vector psi(s,a) (audit / Stage 4)."""
        return float(psi_sa @ self.theta)


@dataclass
class Collective:
    """The warm-started multi-agent collective (Stage 3 output / Stage 4 input)."""

    agents: dict[str, RoleAgent]
    roles: tuple[str, ...]
    normalization: str = ""
    provenance: str = "estimated"  # the policies/rewards are ESTIMATED, not assumed
    meta: dict = field(default_factory=dict)

    @classmethod
    def warm_start(
        cls,
        second_step: SecondStepResult,
        sigma_by_role: dict[str, np.ndarray],
        psi_by_role: dict[str, np.ndarray] | None = None,
        c_by_role: dict[str, np.ndarray] | None = None,
    ) -> "Collective":
        """Build the collective from the Stage-2 result and the Stage-1 CCPs (the warm start)."""
        agents = {}
        for r in second_step.roles:
            agents[r] = RoleAgent(
                role=r,
                sigma=sigma_by_role[r],
                theta=second_step.theta(r),
                psi=None if psi_by_role is None else psi_by_role.get(r),
                c=None if c_by_role is None else c_by_role.get(r),
            )
        return cls(
            agents=agents,
            roles=second_step.roles,
            normalization=second_step.normalization.describe(),
            meta={"lambda": second_step.lam},
        )

    def policy(self, role: str) -> np.ndarray:
        return self.agents[role].sigma

    def rollout(self, transition, horizon: int, start_state: int, seed: int = 0):
        """Roll the collective forward under `transition` (turn-taking DMTA), returning the log.

        Used to produce the Stage-3 collective's behavior for the merit test (evaluation/). Each
        step every role acts on the shared state; the last role's draw advances it. `transition`
        exposes `.prob(s, a)`.
        """
        rng = np.random.default_rng(seed)
        s = int(start_state)
        log = []
        for t in range(horizon):
            for role in self.roles:
                a = self.agents[role].act(s, rng)
                s_next = int(rng.choice(len(transition.prob(s, a)), p=transition.prob(s, a)))
                log.append({"t": t, "role": role, "state": s, "action": a})
                if role == self.roles[-1]:
                    s = s_next
        return log
