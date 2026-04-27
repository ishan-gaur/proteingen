# Models — Agent Notes

Concrete model implementations that subclass the core abstractions.

## Model Registry

Each model lives in its own directory with a dedicated `.md`:

- [esm/esm.md](esm/esm.md) — ESMC, ESM3 (GenerativeModelWithEmbedding subclasses); ESM3IF deprecated (thin subclass of ESM3)
- [dplm2/dplm2.md](dplm2/dplm2.md) — DPLM-2 discrete diffusion protein language model (GenerativeModelWithEmbedding subclass)
- [mpnn/mpnn.md](mpnn/mpnn.md) — ProteinMPNN structure-conditioned sequence design (GenerativeModelWithEmbedding subclass, via `rc-foundry[all]`)
- [frame2seq/frame2seq.md](frame2seq/frame2seq.md) — Frame2seq structure-conditioned inverse folding model (GenerativeModelWithEmbedding subclass, via `frame2seq`)
- [progen3/progen3.md](progen3/progen3.md) — ProGen3 autoregressive protein language model (GenerativeModelWithEmbedding subclass, via `progen3` optional dep)
- [rocklin_ddg/rocklin_ddg.md](rocklin_ddg/rocklin_ddg.md) — Stability predictor from Listgarten lab / ProteinGuide, trained on Rocklin Megascale dataset (PredictiveModel subclass)
- `seki_tyrosine_kinase.py` — kinase fitness predictor (stale, needs updating — sets `self.input_dim` manually)
- [utils.md](utils.md) — structure loading (`load_pdb`, `PDBStructure`) and encoding API for structure-conditioned models. **Read this when adding any structure-conditioned model.**

## Adding a New Model

- **Generative models** (GenerativeModel / GenerativeModelWithEmbedding): use the `/skill:add-generative-model` skill (`.agents/skills/add-generative-model/SKILL.md`) [×1]
- **Predictive models** (PredictiveModel subclasses): use the `/skill:add-predictive-model` skill (`.agents/skills/add-predictive-model/SKILL.md`)
- **Structure-conditioned models**: read [utils.md](utils.md) first — it documents the two-layer API (`PDBStructure` → `atom_array_to_encoding`) and the pattern for writing `condition_from_structure()` / `structure_from_pdb()`. Use the PMPNN implementation as the reference. [×1]
