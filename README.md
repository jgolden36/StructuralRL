# StructuralRL

**Structural initialization for a coordinated multi-agent collective for autonomous-laboratory
discovery.**

Recover the latent primitives (per-role payoffs and beliefs) of human/automated scientific
coordination with a structural dynamic-game estimator, use the recovered quantal-response
equilibrium to *initialize* an agent collective, and only then improve it with reinforcement
learning.

The method and its identification arguments are the two working notes in the repository root —
**these `.tex` files are the source of truth**:

- `structural_init_for_agentic_coordination_v2 (2).tex` — parent method.
- `structural_init_autonomous_labs (1).tex` — companion (autonomous-lab / DMTA) specialization.

See [`CLAUDE.md`](CLAUDE.md) for the full project guidance. This README is a quick start.

## The two phases

- **Phase 1 — verify training is *possible*** (Stages 0–3, no improvement claim). Two tracks:
  - `sim`: recover known payoffs from a synthetic dynamic game (estimator correctness).
  - `tier3`: run on a clean version-controlled computational-discovery corpus (real data path).
- **Phase 2 — train on real lab data** (Tiers 1/2), then Stage 4 in-silico improvement against a
  validated surrogate. *Only if Phase 1 passes the pre-registered go/no-go rule.*

## Pipeline stages

| Stage | What | Module |
|------|------|--------|
| 0 | move taxonomy + state encoder φ | `representation/` |
| 1 | role CCPs σ̂ᵢ + transition F̂ | `estimation/ccp.py`, `transition.py` |
| 2 | partially pooled payoffs θᵢ = θ₀ + δᵢ | `estimation/second_step.py`, `pooling.py`, `forward_sim.py`, `anchor.py`, `identified_set.py` |
| 3 | equilibrium-consistent initialization | `agents/collective.py` |
| 4 | improvement (Phase 2 only) | `agents/improve.py` |

## Install

```bash
pip install -e .            # core estimator (numpy/scipy/pandas)
pip install -e ".[dev]"     # + pytest
pip install -e ".[nn,viz]"  # + torch encoder, matplotlib diagnostics
```

## Run the Phase 1 sim recovery (the canonical correctness check)

```bash
python -m structuralrl.pipelines.phase1_sim --config structuralrl/configs/phase1_sim.yaml
pytest structuralrl/tests/test_recovery.py
```

## Status

Phase 1 `sim` path (synthetic dynamic game → Stages 0–2 → recovery check) is implemented and
tested end-to-end on a tabular logit dynamic-discrete-choice model. Real-data loaders
(`tier3`/`tier1`/`tier2`), the NN encoder, and Stage 4 are scaffolded with documented interfaces
and assumption-labeled stubs, to be filled in per the build order in `CLAUDE.md` §9.

Every estimated/learned/assumed component carries the assumption it leans on in its docstring.
