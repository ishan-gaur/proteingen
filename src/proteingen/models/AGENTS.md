# Models — Agent Notes

Concrete model implementations that subclass the core abstractions.

## Model Registry

Each model lives in its own directory with a dedicated `.md`:

- [esm/esm.md](esm/esm.md) — ESMC, ESM3 (TransitionModelWithEmbedding subclasses); ESM3IF deprecated (thin subclass of ESM3)
- [dplm2/dplm2.md](dplm2/dplm2.md) — DPLM-2 discrete diffusion protein language model (TransitionModelWithEmbedding subclass)
- [mpnn/mpnn.md](mpnn/mpnn.md) — ProteinMPNN structure-conditioned sequence design (TransitionModelWithEmbedding subclass, via `rc-foundry[all]`)
- [rocklin_ddg/rocklin_ddg.md](rocklin_ddg/rocklin_ddg.md) — Stability predictor from Listgarten lab / ProteinGuide, trained on Rocklin Megascale dataset (PredictiveModel subclass)
- `seki_tyrosine_kinase.py` — kinase fitness predictor (stale, needs updating — sets `self.input_dim` manually)
- `utils.py` — shared utilities (`pdb_to_atom37_and_seq`, has TODOs for multi-chain)

## Adding a New Model

- **Generative models** (TransitionModel / TransitionModelWithEmbedding): use the `/skill:add-generative-model` skill (`.agents/skills/add-generative-model/SKILL.md`)
- **Predictive models** (PredictiveModel subclasses): TODO — create an `add-predictive-model` skill
