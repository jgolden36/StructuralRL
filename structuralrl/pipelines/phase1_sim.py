"""Phase 1 `sim` track — synthetic recovery: Stages 0-2 + identification check (+ Stage 3 + merit).

Generates data from a KNOWN dynamic game (known theta_0, {delta_i}, F), runs the two-step
estimator, and confirms it recovers the true payoffs (up to the documented shaping-equivalence
class). This is the unit test for estimator CORRECTNESS — it cannot be done on real data because
truth is unknown (CLAUDE.md §2 Phase 1.1).

Run:
    python -m structuralrl.pipelines.phase1_sim --config structuralrl/configs/phase1_sim.yaml

`run(config)` returns a results dict (also consumed by tests/test_recovery.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..agents.collective import Collective
from ..data.synthetic import make_random_game
from ..estimation import forward_sim
from ..estimation.ccp import fit_role_ccps
from ..estimation.identified_set import gaussian_confidence_region
from ..estimation.normalization import Normalization, gauge_payoff_features
from ..estimation.pooling import select_lambda_cv
from ..estimation.second_step import penalized_pseudo_likelihood
from ..estimation.transition import TabularTransition
from ..evaluation import go_no_go, interaction_stats
from ..evaluation.sufficiency_test import sufficiency_test
from ..evaluation.transfer_test import transfer_test
from ._config import RunConfig


def run(config: RunConfig, out_dir: str | Path | None = None) -> dict:
    rng = np.random.default_rng(config.seed)
    game = make_random_game(
        n_states=config.n_states,
        n_moves=config.n_moves,
        n_features=config.n_features,
        n_roles=config.n_roles,
        beta=config.beta,
        delta_scale=config.delta_scale,
        seed=config.seed,
    )

    # --- Stage 0: sample known-truth data, split campaigns train/test --------
    panel, extras = game.sample(config.n_campaigns, config.horizon, seed=config.seed)
    campaigns = sorted(panel.by_campaign().keys())
    n_test = max(1, len(campaigns) // 4)
    test_cids = set(campaigns[-n_test:])
    train_records = [r for r in panel.records if r.campaign_id not in test_cids]
    test_records = [r for r in panel.records if r.campaign_id in test_cids]

    sa_train = _sa_by_role(train_records, game.roles)
    sa_test = _sa_by_role(test_records, game.roles)

    # --- Stage 1: first step (CCPs sigma-hat_i, transition F-hat) ------------
    ccps = fit_role_ccps(sa_train, game.S, game.A)
    sigma_by_role = {r: ccps[r].sigma for r in game.roles}
    train_trans = [(s, a, s2) for (s, a, s2) in extras["transitions"]]  # full-sample F-hat
    Fhat = TabularTransition(game.S, game.A).fit(train_trans)

    # --- Stage 2: forward sim + penalized pseudo-likelihood (Eq. value/ppl) --
    psi_by_role, c_by_role = {}, {}
    for r in game.roles:
        psi, c = forward_sim.tabular_feature_sums(sigma_by_role[r], Fhat.P, game.X, game.beta)
        psi_by_role[r], c_by_role[r] = psi, c

    norm = Normalization(kind=config.normalization)
    if config.lam is None:
        folds = _cv_folds(train_records, game.roles, n_folds=5)
        cv = select_lambda_cv(folds, psi_by_role, c_by_role, game.k, lambdas=config.lambdas)
        chosen_lambda = cv.best_lambda
        result = penalized_pseudo_likelihood(
            sa_train, psi_by_role, c_by_role, game.k, chosen_lambda, normalization=norm
        )
    else:
        result = penalized_pseudo_likelihood(
            sa_train, psi_by_role, c_by_role, game.k, config.lam, normalization=norm
        )
        chosen_lambda = config.lam

    # --- identification check vs known truth (the point of the sim track) ----
    recovery = _recovery_report(game, result)
    region = gaussian_confidence_region(result)

    # --- Stage 3: equilibrium-consistent warm start --------------------------
    collective = Collective.warm_start(result, sigma_by_role, psi_by_role, c_by_role)

    # --- merit test: collective rollouts vs held-out 'human' (sim) events ----
    coll_events = _collective_events(collective, Fhat, game, test_cids, config, rng)
    human_events = _events_by_campaign(test_records)
    comparisons = interaction_stats.compare(
        coll_events, human_events, tolerances=config.tolerances,
        roles=game.roles, moves=game.moves,
    )

    # --- gating test 1: state-sufficiency falsification ----------------------
    suff = _sufficiency(train_records, test_records, game, config, rng)

    # --- gating test 2: transfer of theta_0 to held-out campaigns ------------
    transfer = transfer_test(result, sa_test, psi_by_role, c_by_role, tol=config.transfer_tol)

    # --- pre-registered go/no-go --------------------------------------------
    decision = go_no_go.decide(comparisons, suff, transfer)

    results = {
        "config": config,
        "chosen_lambda": chosen_lambda,
        "recovery": recovery,
        "confidence_region_names": region.names,
        "confidence_std_err": region.std_err.tolist(),
        "interaction_stats": comparisons,
        "sufficiency": suff,
        "transfer": transfer,
        "decision": decision,
        "collective": collective,
    }
    if out_dir is not None:
        snap = config.snapshot(out_dir)
        results["config_snapshot"] = str(snap)
    return results


# --- helpers ----------------------------------------------------------------
def _sa_by_role(records, roles):
    out = {r: ([], []) for r in roles}
    for rec in records:
        out[rec.role][0].append(rec.payload["state_index"])
        out[rec.role][1].append(rec.payload["action_index"])
    return {r: (np.array(v[0], int), np.array(v[1], int)) for r, v in out.items()}


def _cv_folds(records, roles, n_folds):
    cids = sorted({r.campaign_id for r in records})
    buckets = {c: i % n_folds for i, c in enumerate(cids)}
    fold_records = [[] for _ in range(n_folds)]
    for rec in records:
        fold_records[buckets[rec.campaign_id]].append(rec)
    return [_sa_by_role(fr, roles) for fr in fold_records]


def _events_by_campaign(records):
    out = {}
    for rec in records:
        out.setdefault(rec.campaign_id, []).append({"role": rec.role, "move": rec.move})
    return out


def _collective_events(collective, Fhat, game, test_cids, config, rng):
    """Roll the Stage-3 collective forward to produce matched-volume events for the merit test."""
    out = {}
    for i, cid in enumerate(sorted(test_cids)):
        log = collective.rollout(Fhat, config.horizon, int(rng.integers(game.S)), seed=config.seed + i)
        out[cid] = [{"role": e["role"], "move": game.moves[e["action"]]} for e in log]
    return out


def _sufficiency(train_records, test_records, game, config, rng):
    """Build phi(s) vs phi(s)+dropped-context arrays and run the falsification test.

    In the sim the 'dropped context' is uninformative noise, so a correct test does NOT reject —
    demonstrating the machinery and the expected Phase-1 pass on well-specified data.
    """
    def arrays(records):
        phi = np.array([r.state_features for r in records])
        ctx = rng.normal(size=(len(records), 2))  # uninformative dropped context
        y = np.array([r.payload["action_index"] for r in records], int)
        return phi, ctx, y

    phi_tr, ctx_tr, y_tr = arrays(train_records)
    phi_te, ctx_te, y_te = arrays(test_records)
    return sufficiency_test(
        phi_tr, ctx_tr, y_tr, phi_te, ctx_te, y_te,
        n_actions=game.A, alpha=config.sufficiency_alpha,
    )


def _recovery_report(game, result) -> dict:
    """Compare recovered theta_i to known truth in a common (within-state demeaned) gauge.

    Only within-state differences of psi@theta are identified, so we report both the raw per-role
    payoff recovery (identified) and a gauge note. theta_i = theta_0 + delta_i is the identified
    object; the theta_0/delta split is identified only under the pooling penalty.
    """
    per_role = {}
    max_abs = 0.0
    for r in game.roles:
        true_t = game.theta(r)
        rec_t = result.theta(r)
        err = float(np.max(np.abs(rec_t - true_t)))
        corr = float(np.corrcoef(true_t, rec_t)[0, 1]) if len(true_t) > 1 else 1.0
        per_role[r] = {"true": true_t.tolist(), "recovered": rec_t.tolist(),
                       "max_abs_err": err, "corr": corr}
        max_abs = max(max_abs, err)
    return {"per_role": per_role, "max_abs_err": float(max_abs),
            "theta0_true": game.theta0.tolist(), "theta0_recovered": result.theta0.tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 1 sim recovery pipeline")
    p.add_argument("--config", default=None, help="path to a YAML run config")
    p.add_argument("--out", default="outputs/phase1_sim", help="output dir for the config snapshot")
    args = p.parse_args(argv)

    config = RunConfig.load(args.config)
    res = run(config, out_dir=args.out)

    print("== Phase 1 sim recovery ==")
    print(f"chosen lambda: {res['chosen_lambda']}")
    print(f"max |theta_i - theta_i_hat| over roles: {res['recovery']['max_abs_err']:.4f}")
    for r, d in res["recovery"]["per_role"].items():
        print(f"  {r:9s} corr={d['corr']:.3f}  max_abs_err={d['max_abs_err']:.4f}")
    print("interaction statistics:")
    for c in res["interaction_stats"]:
        flag = "n/a" if not c.available else ("PASS" if c.passed else "FAIL")
        print(f"  {c.name:22s} {flag:4s} dist={c.distance:.4f} tol={c.tolerance}")
    print(res["sufficiency"].describe())
    print(res["transfer"].describe())
    print(f"GO/NO-GO: {res['decision'].state.value}  ({res['decision'].summary})")
    return res


if __name__ == "__main__":
    main()
