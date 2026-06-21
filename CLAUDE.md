# CLAUDE.md — StructuralRL

Guidance for working in this repository. This file describes the code suite needed to
**train and test a coordinated multi-agent collective for autonomous-laboratory discovery**
using *structural initialization*: recover the latent primitives (per-role payoffs and
beliefs) of human/automated scientific coordination with a structural dynamic-game
estimator, use the recovered equilibrium to initialize the agent collective, and only then
improve it with reinforcement learning.

The method and its identification arguments are specified in the two working notes in this
repository:

- `structural_init_for_agentic_coordination_v2 (2).tex` — the **parent** method (general
  pipeline, estimators, normalization theory, power-systems worked example).
- `structural_init_autonomous_labs (1).tex` — the **companion** note specializing the method
  to autonomous laboratories and the design–make–test–analyze (DMTA) loop, including the
  pre-registered proof-of-concept.

**These `.tex` files are the source of truth.** When code and notes disagree, the notes win;
update code (or flag the discrepancy) rather than silently diverging. Keep notation aligned
with the papers ($\phi$, $\hat\sigma_i$, $\hat F$, $\theta_0$, $\delta_i$, $\lambda$, etc.).

---

## 1. What we are building, in one paragraph

A discovery campaign is a collaborative-production game with a clean turn-taking structure
(the DMTA loop). The actors occupy functional **roles** — proposer, designer, executor,
analyst, arbiter. We treat observed coordination as **quantal-response equilibrium play** and
recover, per role, an interpretable flow payoff $\theta_i = \theta_0 + \delta_i$ (a pooled
shared discovery objective plus a shrunken role specialization) using the two-step estimator
of Bajari–Benkard–Levin (BBL) built on the Hotz–Miller inversion. The recovered policies are
mutual best responses by construction, so warm-starting each agent at its recovered policy and
reward places the collective at an *estimate of the human discovery equilibrium*. That is the
auditable baseline; RL improvement (Stage 4) is run later, in silico, anchored to it.

---

## 2. The two-phase deliverable (this is the organizing principle of the suite)

The user-facing goal has two phases. **Build and validate Phase 1 fully before Phase 2.**

### Phase 1 — Verify that training is *possible* (proof of concept, Stages 0–3 only)
Goal: show the machinery runs end-to-end and that the structurally-initialized collective
reproduces held-out interaction *before* any improvement claim. No Stage 4, no real reagents.

Two complementary validation tracks, both required:

1. **Synthetic identification recovery (`sim` track).** Generate data from a *known* dynamic
   game (known $\theta_0$, $\{\delta_i\}$, known $F$). Run Stages 0–2 and confirm the
   estimator recovers the true payoffs (up to the documented shaping-equivalence class) and
   that confidence/identified sets cover truth at the nominal rate. This is the unit test for
   correctness of the estimator itself — it cannot be done on real data because truth is
   unknown.
2. **Clean real-ish corpus (`tier3` track).** Run Stages 0–3 on **Tier 3** data from the
   companion note: a *version-controlled computational-discovery* repository (commits, runs,
   reviews of an in-silico materials/chemistry campaign). This is the cleanest real corpus and
   the discovery analog of the version-controlled coding/writing testbed. It exercises the real
   data path (encoder, taxonomy induction, messy logs) without a wet lab.

Phase 1 succeeds only against the **pre-registered go/no-go rule** (Section 6): the Stage 3
collective matches the named held-out interaction statistics within stated tolerances, the
state-sufficiency falsification test does not reject, and the pooled objective $\theta_0$
transfers to held-out campaigns.

### Phase 2 — Train on real-world laboratory data
Only attempted if Phase 1 passes go/no-go. Choose the corpus with the **estimand** in mind
(parent/companion caveat):

- **Tier 1 — autonomous-platform decision logs.** Richest panel; role-labeled, state-
  conditioned API-call logs from a self-driving lab. *Recovers the platform's encoded
  objective and coordination protocol*, not a scientist's revealed preference.
- **Tier 2 — human-led campaign records.** ELN / LIMS / instrument logs from an instrumented
  human-led high-throughput campaign. *Carries revealed preference*, at the cost of messier
  role labels and more unlogged (tacit) judgment.

Phase 2 runs Stages 0–3 to establish the baseline, then (and only then) Stage 4 improvement
**in silico against a validated surrogate $\hat F$**, under a KL/behavioral anchor to the
Stage 3 policies that preserves coordination structure and **false-discovery discipline** (the
domain's safety-like property). The sim-to-real gap bounds how far improvement may travel from
the explored region; respect it.

---

## 3. Repository layout (target)

The repo currently contains only the two `.tex` notes. The suite below is the intended
structure; create modules as they are implemented and keep this map current.

```
structuralrl/
  data/                  # loaders + schema for each tier; NO raw data committed
    schema.py            #   canonical event record: (campaign_id, t, role, move, payload, state_features, costs)
    synthetic.py         #   Phase 1 `sim`: generate trajectories from a known dynamic game
    tier3_vcs.py         #   Phase 1 `tier3`: parse version-controlled computational-discovery repos
    tier1_platform.py    #   Phase 2: self-driving-lab decision logs
    tier2_human.py       #   Phase 2: ELN/LIMS/instrument logs (human-led)
  representation/
    encoder.py           # Stage 0: state encoder phi(s) -> sufficient-statistic vector
    moves.py             # Stage 0: discrete move taxonomy M (domain spec OR clustering) + validation
  estimation/
    ccp.py               # Stage 1: role-conditional CCPs sigma_i(a|s) (logit/NN)
    transition.py        # Stage 1: transition model F-hat; surrogate/digital-twin adapter
    forward_sim.py       # Stage 2: forward-simulated discounted feature/value sums (Eq. value)
    second_step.py       # Stage 2: penalized pseudo-likelihood (Eq. ppl) + BBL inequality (Eq. pooled)
    pooling.py           # Stage 2: partial pooling theta_i = theta_0 + delta_i; lambda CV selection
    anchor.py            # Stage 2: anchored payoff decomposition pi = c + x^T theta (Eq. anchor)
    identified_set.py    # Stage 2: report identified SETS, not spurious points
    normalization.py     # shaping-equivalence class bookkeeping (Hotz-Miller invariants)
  agents/
    collective.py        # Stage 3: warm-start each role at (sigma_i-hat, theta_0-hat + delta_i-hat)
    improve.py           # Stage 4: multi-agent RL on recovered reward minus lambda_KL * KL(pi || sigma-hat)
  evaluation/
    interaction_stats.py # pre-registered held-out statistics (the merit test)
    sufficiency_test.py  # falsification of state sufficiency (phi vs phi + dropped context)
    transfer_test.py     # does theta_0 transfer across campaigns/labs?
    go_no_go.py          # the pre-registered decision rule
  pipelines/
    phase1_sim.py        # synthetic recovery: Stages 0-2 + identification check
    phase1_tier3.py      # clean corpus: Stages 0-3 + merit test
    phase2_real.py       # Stages 0-3 baseline + Stage 4 improvement on real lab data
  configs/               # one config per run; records normalization, lambda, beta, tolerances
  tests/                 # pytest: estimator unit tests, recovery tests, regression tests
```

---

## 4. The pipeline, stage by stage (what each module must do)

Notation follows the notes. $\phi$ = state encoder; $\hat\sigma_i$ = role CCPs;
$\hat F$ = transition law/surrogate; $\theta_i=\theta_0+\delta_i$ = partially pooled payoffs;
$\beta$ = (calibrated) discount factor; $\lambda$ = pooling strength; $\lambda_{KL}$ = anchor.

- **Stage 0 — Data & representation** (`representation/`, `data/`).
  Fix the discovery move taxonomy $\mathcal{M}$ = {propose, design, execute, characterize,
  accept, reject, replicate, pivot, defer, escalate, stop} (companion note). Validate it by
  predicting held-out moves and check sensitivity to taxonomy granularity. Fit $\phi$ on the
  campaign state — **must include campaign phase / budget-remaining** so the non-stationarity
  is in the state rather than silently absorbed (Assumption: state sufficiency). The realized
  content of a move (specific protocol/compound) is a *within-move structured choice* on top of
  the discrete move type.

- **Stage 1 — First step: policies & transitions** (`estimation/ccp.py`, `transition.py`).
  Estimate role-conditional CCPs $\hat\sigma_i(a\mid s)$ (this is behavioral cloning, read
  structurally) and the transition model $\hat F(s'\mid s,a)$. $\hat F$ is *partly
  deterministic* (queue/accept moves advance the campaign mechanically) and *partly stochastic*
  (nature's response to an experiment). Where a validated surrogate / digital twin exists, it
  supplies $\hat F$ and the Stage 4 rollout environment.

- **Stage 2 — Second step: partially pooled payoffs** (`estimation/second_step.py`,
  `pooling.py`, `anchor.py`, `forward_sim.py`, `identified_set.py`).
  Primary estimator: **penalized pseudo-likelihood** (parent Eq. ppl) — choice-specific values
  by forward simulation against $\hat\sigma_{-i}$ and $\hat F$, implied logit policy, penalized
  by the Gaussian pooling prior. Robustness: **BBL inequality objective** (Eq. pooled) — but
  remember it is *misspecified under quantal play* (bias grows with temperature), so report
  identified **sets** there. Use the **anchored decomposition** $\pi_i = c + x^\top\theta_i$
  (companion Eq. anchor): $c$ collects externally-denominated, *measured* levels — negative
  reagent/instrument cost, negative compute cost & wall-clock, and realized surrogate
  information gain (posterior entropy/variance reduction). These anchors shrink the shaping-
  equivalence class — the **strongest normalization discipline** and a structural advantage of
  this domain. Select $\lambda$ by **cross-validation across held-out campaigns**, scoring on
  fit to held-out interaction statistics (out of sample, not in sample).

- **Stage 3 — Equilibrium-consistent initialization** (`agents/collective.py`).
  For each role: `policy <- sigma_i-hat`, `reward <- theta_0-hat + delta_i-hat`. The collective
  now sits at an estimate of the human discovery equilibrium — the orchestration baseline and
  auditable default. **This stage is fully normalization-invariant** (reproducing the estimated
  policies needs no payoff levels).

- **Stage 4 — Improvement** (`agents/improve.py`) — *Phase 2 only.*
  Multi-agent RL against $\hat F$ on the recovered reward, minus $\lambda_{KL}\cdot
  \mathrm{KL}(\pi_i \,\|\, \hat\sigma_i)$. The anchor preserves coordination and false-discovery
  discipline; it is the structural cousin of offline-RL pessimism. **Normalization caveat:**
  improving one role against frozen rivals is invariant across the equivalence class; *joint*
  movement of all roles is **not** — discipline joint-improvement claims with the Eq.-anchor
  levels, sensitivity across normalizations, and normalization-invariant (externally measured)
  evaluation.

---

## 5. The testing suite (Phase 1's actual deliverable)

`evaluation/` implements the pre-registered merit test from the companion note §5. **Statistics
are named in advance so success cannot be selected after the fact — do not add post-hoc metrics
to a passing run.** The Stage 3 collective is scored on held-out campaigns against:

1. **Role-conditional move frequencies** — how often each role proposes/designs/executes/
   characterizes/replicates/defers.
2. **Move-transition matrix** — which role acts after which, and after which move; does the
   proposer→designer→executor→analyst→arbiter DMTA cadence recur at human rates.
3. **Inter-move & outcome timing** — inter-move intervals, time-to-result,
   experiments-to-discovery distribution.
4. **State-conditional response profiles for pre-registered events** — for a fixed event set
   {anomalous measurement, failed synthesis, replication failure, contradictory result, budget-
   threshold crossing}, does the collective respond at the human rate? Diagnostic actions:
   replicate, escalate, pivot, stop.
5. **Campaign-level outcomes** — experiments-to-validated-discovery, reagent/compute cost per
   validated result, and the **false-discovery / irreproducibility rate** — the tail quantity
   coordination most protects, and **the single most important statistic**.

Plus two gating tests:

- **State-sufficiency falsification** (`sufficiency_test.py`): train a next-move predictor on
  $\phi(s)$ vs. $\phi(s)$ augmented with dropped context (free-text rationale / stated
  hypothesis / prior-campaign memory). A significant predictive gain is **evidence against
  sufficiency** — the binding constraint then becomes instrumentation, not modeling.
- **Transfer of the pooled objective** (`transfer_test.py`): recover payoffs on one set of
  campaigns/labs, test whether they predict coordination on held-out ones, and whether
  $\theta_0$ is the part that transfers. This is the empirical content of "there is a common
  discovery objective to recover at all." Heterogeneity across labs is large — use finite
  mixtures over equilibria or estimate within homogeneous strata; never naively pool labs that
  played different equilibria.

---

## 6. Pre-registered go / no-go rule (`evaluation/go_no_go.py`)

State the decision before running. The approach has merit (and Stage 4 / Phase 2 is worth
running) **iff all three hold**:

1. Stage 3 collective matches the pre-registered interaction statistics within stated tolerances;
2. the state-sufficiency falsification test does **not** reject;
3. the pooled objective $\theta_0$ transfers to held-out campaigns.

Branches: *statistics match but sufficiency rejects hard* → constraint is instrumentation;
prioritize richer logging (rationale/hypothesis) before any improvement claim.
*$\theta_0$ does not transfer* → no common discovery objective across sampled labs; retreat to
homogeneous strata. Encode these outcomes as explicit return states, not free-text.

---

## 7. Identification discipline (do not let the code paper over these)

- **Report identified sets, not points**, wherever weakly identified (inequality estimator,
  equilibrium multiplicity).
- **Discount factor $\beta$ is weakly identified** — calibrate it, never "estimate" it
  silently; report sensitivity of recovered payoffs and downstream behavior to $\beta$.
- **Payoff levels are identified only up to the shaping-equivalence class.** Persist the chosen
  normalization in the run config; Stage 3 is invariant to it, Stage 4 joint improvement is not.
- **Payoff scale ↔ logit scale are not separately identified.** The shock scale is normalized to
  one; recovered payoffs are in shock units; the LLM sampling temperature inherits this — cross-
  lab temperature comparisons are statements about *relative* payoff scale.
- **Off-path beliefs are not identified from on-path data.** The recovered model is trustworthy
  near the human equilibrium and increasingly extrapolative away — the binding risk for Stage 4.
- **Surrogate $\hat F$ is reliable only near the explored region.** The sim-to-real gap bounds
  off-path improvement; flag rollouts that leave the surrogate's validated support.

---

## 8. Engineering conventions

- **Language/stack:** Python. Suggested: PyTorch (encoder, CCPs, RL), NumPy/SciPy + a
  numerical optimizer for the second step, `pandas`/`pyarrow` for panels, `hydra`/dataclass
  configs, `pytest` for tests, `matplotlib` for diagnostics. (No stack is committed yet — pick
  these unless a later decision overrides, and record the choice in `pyproject.toml`.)
- **Determinism:** every pipeline run takes a seed and writes a config snapshot recording
  normalization, $\lambda$, $\beta$, tolerances, and corpus/tier. Recovery and merit-test runs
  must be reproducible.
- **No fabricated data.** Synthetic data exists only in `data/synthetic.py` for the Phase 1
  `sim` recovery track and is always labeled as such. Never synthesize "results."
- **Data governance:** raw lab/platform data is **never committed** (size, licensing, possible
  confidentiality). `data/` holds loaders and schema; real data is referenced by path/config.
- **Tests first for the estimator:** the synthetic recovery test (`tests/`) is the canonical
  correctness check — it must pass before any real-data run is trusted.
- **Labeling learned vs. estimated vs. assumed:** keep the paper's discipline — every component
  carries the assumption it leans on; surface assumptions in docstrings.

---

## 9. Suggested build order

1. `data/schema.py` + `data/synthetic.py` — canonical record and a known-truth generator.
2. `representation/moves.py`, `representation/encoder.py` — taxonomy + $\phi$ (with phase/budget).
3. `estimation/ccp.py`, `estimation/transition.py` — Stage 1.
4. `estimation/forward_sim.py`, `second_step.py`, `pooling.py`, `anchor.py`,
   `identified_set.py` — Stage 2; validate against synthetic truth.
5. `agents/collective.py` — Stage 3.
6. `evaluation/*` + `pipelines/phase1_sim.py`, `phase1_tier3.py` — the merit test and go/no-go.
7. **Gate:** pass go/no-go on Tier 3 before touching real lab data.
8. `data/tier1_platform.py` / `tier2_human.py`, `agents/improve.py`,
   `pipelines/phase2_real.py` — Phase 2.

---

## 10. Git / workflow

- Develop on branch **`claude/quirky-dijkstra-1gzruh`**; create it locally if absent.
- Commit with clear, descriptive messages; push with `git push -u origin <branch>` (retry with
  exponential backoff on network errors only).
- Do **not** open a pull request unless explicitly asked.
</content>
</invoke>
