# Documentation Site Plan — ProteinGen

## Stack

- **mkdocs-material** (already installed)
- **mkdocstrings[python]** (already installed)
- Hosted via GitHub Pages at `ishan-gaur.github.io/proteingen/`

## Navigation

| Tab            | Status     | Notes |
|----------------|------------|-------|
| **Home**       | Scaffold   | Problem statement, design philosophy blurb (agent-forward, wetlab workflows, drylab contributing) |
| **Setup**      | Scaffold   | Claude Code install, uv setup, repo cloning, pointer to Design Philosophy |
| **Examples**   | Scaffold   | Walk through existing `examples/` scripts |
| **Models**     | Scaffold   | Table of models: ESMC, ESM3, ESM3IF, StabilityPMPNN; links, weights, conditioning vars, logit format |
| **Workflows**  | Scaffold   | ProteinGuide recipe tabs; each step refs a sub-workflow |
| **Contributing** | Scaffold | How to prompt an agent to add a new model; skill files for generative + predictive model migration |
| **Reference**  | Scaffold   | Auto-generated API docs via mkdocstrings |

---

## Page Details

### Home (`docs/index.md`)
- 3–4 sentences: guided protein sequence design, discrete generative models + predictive models via Bayes' rule.
- Agent-forward: designed so wetlab scientists can use it via AI agents (Claude Code), with extensive workflow guides and evaluation prompts.
- Drylab scientists: contribute models, use included skill files for properly-organized PRs.
- Note: "coming soon" for features not yet implemented.

### Setup (`docs/setup.md`)
1. Install Claude Code (link to official docs)
2. Install uv (link to astral docs)
3. Create a project repo; have Claude clone ProteinGen and add it to AGENTS.md
4. Command snippet: `git clone ... && echo "..." >> AGENTS.md`
5. Pointer: "Read Design Philosophy in the Reference tab to learn about the three base classes"
6. Quick examples of `set_temp_()`, `conditioned_on()`, etc.
- TODO[pi]: finalize the "recommended project setup" flow

### Examples (`docs/examples.md`)
Walk through each existing example:
1. `unconditional_sampling.py` — simplest: ESMC + `sample_any_order_ancestral`
2. `esm3_structure_conditioned_sampling.py` — ESM3 with atom37 coords
3. `pca_embedding_init.py` — PCA-init EmbeddingMLP from ESMC
4. `trpb_linear_probe.py` — train a probe, do guided sampling
5. `stability_guidance/` — WIP, "coming soon"

### Models (`docs/models.md`)
Table columns: Model | Source | Weights | Conditioning Variables | Output Format
- ESMC (300m/600m) — github.com/evolutionaryscale/esm, HF `esmc_300m`/`esmc_600m`, none (masked LM), `(B, L, 64)` logits
- ESM3 — same repo, HF `esm3-open`, structure (atom37 coords), `(B, L, 64)` logits
- ESM3IF — same repo, HF `esm3-open`, structure (atom37 coords, required), `(B, L, 64)` logits
- StabilityPMPNN — rocklin_ddg in-repo, manual weights, structure (PDB → featurize), scalar stability logit

TODO[pi]: figure out model storage strategy — convert to HF format? Use torch hub cache as default (like evodiff/zenodo)? Extend `save/from_checkpoint` interface? Needs design decision.

### Workflows (`docs/workflows/`)
Subpages (or tabs within a single page):

**ProteinGuide** (`docs/workflows/protein-guide.md`):
1. Organize your data → ref to data workflow (TODO[pi]: write this workflow)
2. Set up a PredictiveModel subclass → ref to predictive model workflow (TODO[pi])
3. Train a noisy classifier → ref to training workflow (TODO[pi])
4. Combine with TAG or DEG → ref to guidance workflow (TODO[pi])
5. Sample: linear interpolator, euler, or ancestral → ref to sampling workflow (TODO[pi])

Each step is a "recipe" card linking to a more detailed sub-workflow.
For now, all sub-workflows are "coming soon" stubs.

### Contributing (`docs/contributing.md`)
- How to prompt an agent to add a new model (generative or predictive)
- Skill files in `contributing/` folder of repo (TODO[pi]: create these)
  - `contributing/add_generative_model.md` — migration checklist, tests, HF upload, docs changes
  - `contributing/add_predictive_model.md` — same for predictive models
- Link skill files from AGENTS.md so agents discover them automatically
- TODO[pi]: write the actual skill files

### Reference (`docs/reference/`)
Two parts:
1. **Design Philosophy** (`docs/reference/design-philosophy.md`) — expanded version of README's "Core Abstractions" section. Covers:
   - ProbabilityModel (base class, temperature, conditioning)
   - TransitionModel (generative model wrapper, LoRA)
   - PredictiveModel (binary logit pattern, target management, gradients)
   - Guidance (TAG/DEG, Bayes' rule combination)
   - Sampling infrastructure
2. **API Reference** — auto-generated via mkdocstrings:
   - `docs/reference/probability_model.md` → `::: dfm.probability_model`
   - `docs/reference/generative_modeling.md` → `::: dfm.generative_modeling`
   - `docs/reference/predictive_modeling.md` → `::: dfm.predictive_modeling`
   - `docs/reference/guide.md` → `::: dfm.guide`
   - `docs/reference/sampling.md` → `::: dfm.sampling`
   - `docs/reference/data.md` → `::: dfm.data`
   - `docs/reference/models.md` → `::: dfm.models`

---

## Minimal Implementation (this session)

1. ✅ Write this PLAN.md
2. Write `mkdocs.yml` with full nav, material theme config, mkdocstrings plugin
3. Create all page stubs with correct content / "coming soon" markers
4. Add TODO[pi] comments to README for unfinished items
5. Verify `uv run mkdocs build` succeeds
6. Update `.gitignore` to exclude `site/`

## NOT this session

- Actually write the workflow sub-pages in detail
- Create the contributing skill files
- Polish the home page design
- Add custom CSS / branding
- Set up GitHub Actions for auto-deploy
- Resolve stale imports in guide.py / sampling.py
