# Models

## Generative Models

| Model | Class | Source | Conditioning | Output |
|-------|-------|--------|-------------|--------|
| [ProteinMPNN](proteinmpnn.md) | `proteingen.models.ProteinMPNN` | [Foundry](https://github.com/dauparas/ProteinMPNN) (via `rc-foundry[all]`) | Structure (required) | `(B, L, 22)` logits |
| LigandMPNN | *coming soon* | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN) | Structure + ligands | Sequence logits |
| [ESMC](esmc.md) (300m/600m) | `proteingen.models.ESMC` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | None (masked LM) | `(B, L, 64)` logits |
| [ESM3](esm3.md) | `proteingen.models.ESM3` | [EvolutionaryScale/esm](https://github.com/evolutionaryscale/esm) | Structure (atom37 coords) | `(B, L, 64)` logits |
| [ESM Forge API](esm-forge-api.md) | `proteingen.models.ESMForgeAPI` | [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) | Structure (ESM3 only) | `(B, L, 64)` logits |
| [DPLM-2](dplm2.md) | `proteingen.models.DPLM2` | [bytedance/dplm](https://github.com/bytedance/dplm) | None (masked diffusion) | `(B, L, 8229)` logits |
| [Frame2seq](frame2seq.md) | `proteingen.models.Frame2seq` | [dakpinaroglu/Frame2seq](https://github.com/dakpinaroglu/Frame2seq) | Structure (`pdb_path`, `chain_id`) | `(B, L, 22)` logits |
| Dayhoff | *coming soon* | [microsoft/Dayhoff](https://huggingface.co/microsoft/Dayhoff-170m-UR50) | None (masked LM) | Sequence logits |
| [ProGen3](progen3.md) | `proteingen.models.ProGen3` | [Profluent-AI/progen3](https://github.com/Profluent-AI/progen3) | None (autoregressive LM) | `(B, L, 134)` logits |
| EvoDiff | *coming soon* | [microsoft/evodiff](https://github.com/microsoft/evodiff) | None (discrete diffusion) | Sequence logits |
| SaProt | *coming soon* | [westlake-repl/SaProt](https://github.com/westlake-repl/SaProt) | Structure (Foldseek tokens) | Sequence logits |
| AMPLIFY | *coming soon* | [chandar-lab/AMPLIFY](https://github.com/chandar-lab/AMPLIFY) | None (masked LM) | Sequence logits |

## Predictive Models

| Model | Class | Source | Conditioning | Output |
|-------|-------|--------|-------------|--------|
| [StabilityPMPNN](stability-pmpnn.md) | `proteingen.models.rocklin_ddg.PreTrainedStabilityPredictor` | [ProteinGuide](https://arxiv.org/abs/2505.04823) (ProteinMPNN-based) | Structure (PDB → featurize) | Scalar stability logit |
| METL | *coming soon* | [gelman-lab/METL](https://github.com/gelman-lab/METL) | Structure (biophysics-pretrained) | Fitness scalar |
| Tranception | *coming soon* | [OATML-Markslab/Tranception](https://github.com/OATML-Markslab/Tranception) | MSA (retrieval-augmented) | Fitness scalar |
