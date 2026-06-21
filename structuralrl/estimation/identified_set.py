"""Stage 2 — report identified SETS, not spurious points (CLAUDE.md §7).

Wherever payoffs are weakly identified (the BBL inequality estimator; equilibrium multiplicity;
the weakly-identified discount factor beta), the honest object is a SET. Two routines:

* :func:`gaussian_confidence_region` — for the penalized pseudo-likelihood (a smooth M-estimator),
  an asymptotic ellipsoid from the inverse Hessian of the negative objective. Reports per-
  coordinate standard errors and a coverage helper used by the recovery test to check the nominal
  rate.
* :func:`inequality_identified_set` — for the BBL inequality objective (Eq. pooled), the set of
  theta whose objective is within a tolerance of the minimum (a level set), sampled on a grid /
  by random search. This is the correct partial-identification report under quantal-play
  misspecification.

Caveat carried in docstrings: levels are identified only up to the shaping class; these sets are
*conditional on the recorded normalization* (normalization.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .second_step import SecondStepResult


@dataclass
class ConfidenceRegion:
    mean: np.ndarray
    cov: np.ndarray
    names: list[str]

    @property
    def std_err(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, np.inf))

    def contains(self, point: np.ndarray, level: float = 0.95) -> bool:
        """Whether `point` lies in the `level` ellipsoid (Mahalanobis / chi-square)."""
        from scipy.stats import chi2

        d = np.asarray(point, float) - self.mean
        try:
            m2 = float(d @ np.linalg.solve(self.cov, d))
        except np.linalg.LinAlgError:
            return False
        return m2 <= chi2.ppf(level, df=len(self.mean))


def gaussian_confidence_region(result: SecondStepResult) -> ConfidenceRegion:
    """Asymptotic confidence region for (theta0, {delta_i}) from the inverse Hessian.

    For an M-estimator minimising the negative penalized pseudo-likelihood, the inverse Hessian
    at the optimum approximates the parameter covariance (a sandwich form would be more robust;
    this is the model-based plug-in). Weakly-identified directions surface as large variances —
    the signal to widen to a set rather than trust a point.
    """
    if result.hessian is None:
        raise ValueError("SecondStepResult has no Hessian; rerun second step to populate it.")
    k = len(result.theta0)
    roles = result.roles
    names = [f"theta0_{j}" for j in range(k)]
    for r in roles:
        names += [f"delta[{r}]_{j}" for j in range(k)]
    mean = np.concatenate([result.theta0] + [result.deltas[r] for r in roles])
    cov = np.linalg.pinv(result.hessian)
    return ConfidenceRegion(mean=mean, cov=cov, names=names)


def inequality_identified_set(
    objective,
    theta_grid: np.ndarray,
    tol: float = 1e-2,
) -> np.ndarray:
    """Level-set identified region for the BBL inequality objective (Eq. pooled).

    `objective(theta) -> float`; `theta_grid` is (G, k) candidate points. Returns the subset of
    grid points whose objective is within `tol` of the observed minimum — the identified set
    under deterministic-payoff (inequality) misspecification. Report this, not its argmin.
    """
    vals = np.array([objective(t) for t in theta_grid])
    qmin = vals.min()
    return theta_grid[vals <= qmin + tol]


def beta_sensitivity(recover_fn, betas) -> dict[float, SecondStepResult]:
    """Report sensitivity of recovered payoffs to the (weakly identified) discount factor beta.

    beta is calibrated, never silently estimated (CLAUDE.md §7). `recover_fn(beta)` reruns Stage 2
    at a given beta; the spread of returned payoffs across `betas` is the sensitivity to report.
    """
    return {float(b): recover_fn(float(b)) for b in betas}
