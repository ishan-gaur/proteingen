# Recursive Modules Documentation Refactor — Work Log

## Task Summary
Reorganize ProteinGen documentation around the four-module principle (Data, Models, Sampling, Evaluation) as the organizing abstraction. Workflows are named compositions of modules (like papers: ProteinGuide, CbAS, etc.). Examples are concrete instantiations. Modules are reusable mathematical implementations (TAG/DEG = Bayes' rule, likelihood curves = evaluation tool, etc.).

## Key References
- Homepage with four modules: `docs/index.md` (bottom section "Library Design with ProteinGen")
- PbrR walkthrough (ProteinGuide impl): `~/dfm-worktrees/pbrr-walkthrough/examples/pbrr_walkthrough/`
- ProteinGuide user guide notes: `proteinguide-userguide-high-level-draft.md`
- Stability guidance example: `examples/stability_guidance/main.py` (uses `sample_ctmc_linear_interpolation`, TAG, ESM3)

## The Four Modules
1. **Data**: ProteinDataset, noise functions, time samplers, FASTA utils, MSA tools, data splits
2. **Models**: GenerativeModel (ESM3, ESMC, DPLM2, PMPNN), PredictiveModel (probes, MLPs, XGBoost), guidance (TAG, DEG), training (LoRA, noisy classifier training)
3. **Sampling**: sample (discrete-time ancestral), sample_ctmc_linear_interpolation, sample_flow_matching_legacy
4. **Evaluation**: likelihood curves, oracle scoring, predictor-oracle agreement, diversity metrics, seq identity (stubs needed for most)

## Workflows = Named Compositions from Papers
- ProteinGuide: data + finetune base model + train oracle + train noisy predictor + validate agreement + guided sampling + evaluate
- CbAS (David Brookes): data + generative model + predictor + importance-weighted retraining
- ProteinDPO: data + preference pairs + DPO training of generative model
- MLDE: directed evolution, MCMC in sequence space
- DADO (James Bowden): data-augmented directed optimization
- Continued Pretraining: MSA data + LoRA finetuning + likelihood eval

## Steps & Status
- [x] Step 0: Read all existing docs, pbrr example, user guide notes
- [x] Step 1: Complete homepage final section (unconditional example + ProteinGuide first-round sketch) → commit de1db2e
- [x] Step 2: Reorganize API reference around four modules → commit 4f3f63b
- [x] Step 3: Reorganize workflows section (separate workflows from modules) → commit 8c88616
- [x] Step 4: Add recursive module breakdown at top of each example → commit d81f435
- [x] Step 5: Iterate on abstractions, refine → commit 88894da
- [x] Step 6: Create workflow-following skill → commit 982b2b4

## Stubs Needed (track for reporting back)
- `docs/reference/evaluation.md` — likelihood curves exists, need stubs for: oracle scoring, predictor-oracle agreement, diversity metrics, mutational distance, activity range analysis
- Data splits by mutational distance / activity range / position (mentioned in user guide notes)
- MSA acquisition tools/links (Steinegger lab: MMseqs2, Foldseek for structure-based)
- Sensitivity analysis utilities

## Decisions Made
- Evaluation is mostly aspirational; add stubs but don't go overboard
- Workflows at paper-level conceptual scope; modules are the shared reusable pieces
- Homepage ProteinGuide sketch = annotated code for a basic first round (not full user guide)
- Recursive decomposition in examples should be approachable, not too deep (Ousterhout: deep core, simple interface)
