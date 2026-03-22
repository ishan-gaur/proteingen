# Models — Agent Notes

Concrete model implementations that subclass the core abstractions.

## Model Registry

Each model lives in its own directory with a dedicated `.md`:

- [esm/esm.md](esm/esm.md) — ESMC, ESM3, ESM3IF (TransitionModelWithEmbedding subclasses)
- [rocklin_ddg/rocklin_ddg.md](rocklin_ddg/rocklin_ddg.md) — Stability predictor (PredictiveModel subclass)
- `seki_tyrosine_kinase.py` — kinase fitness predictor (stale, needs updating — sets `self.input_dim` manually)
- `utils.py` — shared utilities (`pdb_to_atom37_and_seq`, has TODOs for multi-chain)

## Adding a New Model

- **Generative models** (TransitionModel / TransitionModelWithEmbedding): use the `/skill:add-generative-model` skill (`.agents/skills/add-generative-model/SKILL.md`)
- **Predictive models** (PredictiveModel subclasses): TODO — create an `add-predictive-model` skill
