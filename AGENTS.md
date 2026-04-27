# Agent Guidelines

## Skills

Available skills:
- `follow-workflow` — plan and implement a library design pipeline by walking through workflows step-by-step
- `add-generative-model` — workflow for integrating a new generative model into the library
- `add-predictive-model` — workflow for integrating a new predictive model into the library
- `likelihood-curves` — evaluate and plot log-likelihood trajectories for generative models under progressive unmasking. Also supports teacher-forced decode trajectories via `compute_decoding_log_prob_trajectory`.

## Project Management

- Use `uv` for all package management and running Python code [×17]
  - Install dependencies: `uv add <package>`
  - Run scripts: `uv run python <script>`
  - Sync environment: `uv sync`
  - Install package in editable mode: `uv pip install -e .`
  - Run tests: `uv run python -m pytest tests/ -v`
  - Run formatter: `uv run ruff format`
  - Run lineter: `uv run ruff check`
- Use this file to note down project-related info important to know across sessions
- **Discuss design decisions before implementing** - especially for abstractions and class structures [×1]
- **Ask for clarification when instructions seem contradictory** - don't guess intent, surface the confusion

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and modular
- **Tensor shape annotations on every intermediate variable** — every tensor assignment in model/pipeline code should have an inline comment with shape and meaning, e.g. `# h_EXV [B, L, K, 3H] - encoder edge+node features masked by anti-causal`. Define an index legend at the top of each model class (e.g. `S: batch, P: position, T: token dim, D: embedding dim, K: neighbors, H: hidden`) and reference those letters in all shape comments. This is the single most effective thing for making tensor code readable — learned from Foundry's MPNN implementation. [×1]
- Make user-provided inputs required, not optional - if the user needs to provide something, the interface should demand it
- Prefer strict interfaces that prevent misuse over permissive ones that raise errors at runtime
- Let code crash naturally rather than wrapping in try/catch - silent failures hide bugs
- Use assert statements to validate inputs/outputs in complex pipelines and tricky functions - they serve as executable documentation and catch issues early
- Follow John Ousterhout's "A Philosophy of Software Design": simple interfaces with deep implementation, avoid shallow classes
- Use `raise ValueError` for input validation at API boundaries, asserts for internal invariants
- Prefer torch.Tensor over numpy arrays when data will be used with models, training, or inference.
- Keep return structures flat and minimal - only include what's actually needed; if they get more complicated, define a typeddict to provide self-documenting structure
- Implement functionality or remove it - no silent no-ops (e.g. don't accept a parameter and ignore it)

## Project Structure

- `src/proteingen/` — core library, installed as editable package (`uv pip install -e .`). See → [src/proteingen/AGENTS.md](src/proteingen/AGENTS.md)
  - Per-component design docs co-located with source (e.g. `probability_model.md` next to `probability_model.py`)
  - `models/` — each model in its own subdirectory with a `.md` (e.g. `models/esm/`, `models/rocklin_ddg/`)
- `examples/` — end-to-end usage examples (sampling, guidance, probes, PCA init). See → [examples/AGENTS.md](examples/AGENTS.md)
- `tests/` — pytest suite (current status: 318/319 passing, 1 pre-existing failure in `test_esmc_temperature`). See → [tests/AGENTS.md](tests/AGENTS.md)
- `scripts/` — one-off experiment scripts (embedding computation, guidance experiments)
- `docs/` — MkDocs documentation site. See → [docs/AGENTS.md](docs/AGENTS.md)
- `TODO.md`, `PLAN.md`, `DESIGN.md` — roadmap and design docs


## Benchmark Model Families Notes

- `examples/benchmark_model_families/prepare_data.py` currently queries `(reviewed:true) AND (length:[min TO max])` without an organism diversity constraint; with seed 42 this returns all-human sequences. If taxonomic diversity matters, add an explicit organism filter/sampling strategy.
- Teacher-forced trajectories are saved per model at `outputs/<MODEL>/teacher_forced_trajectory.json` and plotted by `analyze.py` as `teacher_forced_likelihood_trajectories.png`.
- Teacher-forced trajectory computation is the slow part of `generate.py` (~5-50s per sequence on GPU depending on length). It batches `batch_size=32` decode steps per forward pass.
- Running all 6 models on 4 GPUs in parallel via backgrounded `CUDA_VISIBLE_DEVICES` processes works well. Each model only needs ~2-8 GB VRAM for 20 sequences of length 80-300.
- With 20 sequences × 1 order × 4 masking levels = 80 generation samples per model, 480 total.

## AF3 Server

- **Separate repo**: `~/af3-server/` (GitHub: `ishan-gaur/af3-server`, public) — FastAPI server + Python client for persistent AF3 inference [×3]
- Installed from GitHub: `af3-server @ git+https://github.com/ishan-gaur/af3-server.git`
- Import: `from af3_server import AF3Client`
- Package structure: `src/af3_server/` (client, pip-installable), `server/` (server.py + .def, runs inside container)
- **Server's real value is cross-environment access** — AF3 runs in JAX/Apptainer container, proteingen code runs in PyTorch env. The HTTP boundary bridges them. For batch-only workflows, a simpler in-container script would suffice. [×1]
- **Official AF3 codebase is a full Python library** — `ModelRunner`, `predict_structure()`, `folding_input.Input` etc. are all importable, not just CLI. Our server wraps these same functions.
- **Server config**: `num_diffusion_samples` (default 5) and `num_recycles` (default 10) are per-server env vars (`AF3_NUM_DIFFUSION_SAMPLES`, `AF3_NUM_RECYCLES`), NOT per-request — model config is set once at startup
- **Single GPU only** — processes jobs sequentially. For multi-GPU, run multiple server instances on different ports/GPUs.
- **Apptainer** installed at user level: `~/bin/apptainer` v1.4.5, `~/lib/libfuse3.so.3`, `~/libexec/apptainer/bin/squashfuse_ll` (wrapper for libfuse3)
- **SIF image**: `/data/apptainer_images/alphafold3_server.sif` (alphafold3 Docker image + FastAPI/uvicorn)
- **Model weights**: `/data/af3/af3.bin`
- **Databases**: `/data/af3db/` (original), `/data/af3db_updated/` (newer PDB from 2025/11/14) [×1]
- **RTX 6000 Ada (49GB) requires unified memory**: `TF_FORCE_UNIFIED_MEMORY=true`, `XLA_CLIENT_MEM_FRACTION=3.2` — OOMs with `XLA_PYTHON_CLIENT_PREALLOCATE=true`
- **Installed AF3 package differs from source repo**: container uses `alphafold3.model.model.Model` (not `diffusion_model.Diffuser`), `extract_inference_results` (not `extract_structures`)
- **JAX compilation cache**: `/data/af3_jax_cache` — persists compiled models across server restarts
- hanlun runs AF3 via Docker with `--norun_data_pipeline` (pre-computed MSA); their SLURM array jobs use all 4 GPUs
- `launch.sh` can be submitted on Kraken with explicit account/QoS flags: `sbatch --cluster=kraken --account=researchers --qos=high-prio --export=ALL,AF3_PORT=<port>,AF3_DB_DIR=/data/af3db_updated launch.sh`; server binds `0.0.0.0` and is reachable on the machine's public IP at that port.
- Even when bound to `0.0.0.0` and healthy locally, off-machine internet access can still time out due to upstream campus/cloud firewall policies; SSH local port forwarding (`ssh -L <local_port>:localhost:<remote_port> user@host`) is a reliable fallback.

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `peft>=0.13.0` for LoRA adapter support
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — harmless
- **Repo renamed to `proteingen`** — display name is **ProteinGen**, URLs/paths/package-slug stay lowercase `proteingen`
- Git remote: `git@github.com:ishan-gaur/proteingen.git`

## Optional Dependency Extras

- `proteingen[pmpnn]` → `rc-foundry[all]`, `proteingen[af3]` → `af3-server`, `proteingen[all]` → both
- `models/__init__.py` guards ProteinMPNN and PreTrainedStabilityPredictor imports with `try/except ImportError` so base `import proteingen` works without optional extras
- ProteinMPNN weights auto-download on first use via `foundry_cli.download_checkpoints.install_model` — no manual `foundry install proteinmpnn` step needed
- `af3-server` is not imported anywhere in proteingen source — purely a convenience dependency for users
- `mkdocs-liveedit` and `af3-server` both resolve via `[tool.uv.sources]` git URLs — pip users without uv can't resolve these from PyPI (af3-server: `pip install git+https://github.com/ishan-gaur/af3-server.git`)
- Dev/docs dependencies (mkdocs, ruff, pytest, python-lsp-server) stay in base package — users are contributors by default

## Docs Conventions

- Setup docs use tab pairs: always show both "uv" and "conda / pip" paths
- Conda tab recommends Miniforge (ships with mamba) but calls everything "conda" to avoid confusing users
- `??? note` admonition = collapsed by default (used for Installing Claude Code section)

## ProteinDataset Design Decisions

Replaces old `GuidanceDataset`. Key design choices from discussion:

- **Dataset holds raw data only** — sequences, observations, labels. All model-specific transforms (tokenization, noising, padding) happen in the collator.
- **Collator is model-specific** — `dataset.collator(model, noise_fn, time_sampler)` returns a collate_fn. The model provides tokenizer and `preprocess_observations`.
- **`noise_fn` and `time_sampler` are required, not optional** — use sentinels `no_noise` + `fully_unmasked` for clean training. This forces the user to be explicit.
- **`NoiseFn = (input_ids, t) -> noised_ids`** — owns the corruption strategy. `TimeSampler = () -> float` — owns when/how much. Separated so you can reuse the same corruption with different `t` distributions.
- **Observations, not conditioning** — dataset stores "observations" (what the model sees). No "conditioning → observations" rename layer inside the dataset.
- **`rename_obs_keys` on the collator** — `{model_kwarg: dataset_key}` for when two models use different names for the same data. One dataset, multiple collators.
- **`preprocess_observations` should be batched** — called per-batch in the collator with list-valued dicts. Not yet fully implemented on the model side (ESM3's `preprocess_observations` is still single-sample).
- **Training uses `model(input_ids, **observations)` not `get_log_probs`** — `get_log_probs` adds log_softmax/temp which isn't wanted for training loss.
- **Loss on masked positions only** — `F.cross_entropy(logits[masked], target[masked])` where `masked = input_ids != target_ids`. Unmasked positions have trivial loss due to logit formatting.

## ESM3 Training Gotchas

- **ESM3 fp32 logits overflow** — without AMP (bfloat16), loss can be `inf`. Always use `--amp` for GPU training.
- **ESM3 structure conditioning requires fixed-length sequences** — structure tokens are (L+2,) with BOS/EOS. All sequences in the batch must match the structure length. For MSA fine-tuning with variable-length sequences, don't use structure conditioning.
- **ESM3 `forward()` doesn't use `self.observations`** — only `get_log_probs` bridges `set_condition_` → `collate_observations` → `forward(**obs)`. For training, manually call `model.collate_observations(input_ids, model.observations)` then `model(input_ids, **obs)`.
- **ESM3 VQ-VAE lazy loading** — `set_condition_` triggers VQ-VAE encoder load (~30M params). Must re-freeze all non-LoRA params after this call.
- **SLURM output buffering** — Python's stdout is block-buffered when redirected to a file. Use `sys.stdout.reconfigure(line_buffering=True)` at the start of `main()` for SLURM log visibility.

## EphB1 MSA Data Notes

- `examples/finetune_esm3/EphB1_MSA.fasta` — 14,335 UniRef sequences aligned to the EphB1 kinase domain (295 residues, UniProt P54762 residues 602-896)
- **Sequences are domain fragments, not full-length proteins** — MSA headers contain alignment coordinates (e.g. `560 854 943` = hit residues 560-854 of a 943-residue protein). After gap removal you get ~200-295 residue kinase domains.
- After gap removal and filtering ≥200 residues: ~9,947 sequences
- Only 16 sequences are exactly 295 residues (required for structure conditioning)
- Characters: standard 20 AAs + X (106 sequences have X), gap chars `-`
- `7KPM_atom37_295.pt` — preprocessed atom37 coords (295, 37, 3) for ESM3. Same file as in `kortemme_tyrosine_kinase_design/structures/`. Built by renaming PTR→TYR, filtering ADP ligand.
- `pdb_to_atom37_and_seq()` crashes on raw 7KPM.pdb due to PTR (phosphotyrosine) and ADP ligand — use the preprocessed .pt file.

## Worktree Cleanup Notes

- `dfm-worktrees/` may contain orphaned directories not tracked by `git worktree list` — check with `ls` not just `git worktree list`
- If a worktree folder was manually deleted, run `git worktree prune --verbose` in the main repo to remove stale registrations (`git worktree remove` cannot run if the path is already gone).
- Current registered worktrees: `main`, `spawn/pbrr-walkthrough`, `spawn/progen`, `spawn/recursive-modules`, `docs`, `spawn/pmpnn`, `spawn/gaussian-predictors`.
- Deleted branches that still have stale local/remote refs: `spawn/cinderdrella`, `spawn/likelihood-curves` — can be cleaned with `git branch -d` / `git push origin --delete`
- One stash exists on `spawn/landing-page` branch: `WIP on spawn/landing-page: 1d4d90d`
- `dfm-worktrees/landing-page-copy/` also exists — status unknown

## AlphaFold 3 Setup

- Docker image `alphafold3:latest` available on this machine
- Model weights: `/data/af3/af3.bin`
- Databases: `/data/af3db/` (original), `/data/af3db_updated/` (newer PDB from 2025/11)
- Source: `/home/ishan/local_dependencies/alphafold3/`
- Run pattern: `docker run -v /data/af3:/app/models -v /data/af3db:/public_databases -v <input>:/app/af_input -v <output>:/app/af_output alphafold3 python run_alphafold.py --json_path=... --model_dir=/app/models --output_dir=/app/af_output`
- `--norun_data_pipeline` flag skips MSA search (for pre-computed MSAs)
- User hanlun runs it regularly; last successful run was via docker

## Refactoring Tools

- **Jedi for Python renames**: `jedi` is installed in the project. Use `jedi.Script(path=..., project=jedi.Project('.')).rename(line, col, new_name=...)` for project-wide symbol renames (imports, type hints, inheritance). Call `.apply()` to write changes. Handles Python symbols only — docstrings, comments, `__all__` entries, and markdown must be updated separately with sed.
- **Rename ordering**: when renaming `Foo` and `FooBar`, always rename the longer name first to avoid partial matches with sed.
- **Test temperature on masked positions**: logit formatting makes non-mask positions one-hot, so temperature scaling is a no-op there. Temperature tests must use mask tokens to see any effect.

## Sampling API

- `sample()` is the unified sampler — replaces old `sample_any_order` and `sample_in_order`. Returns `SamplingTrajectory` (TypedDict with `sequences`, `step_log_probs`, `step_positions`, `step_tokens`).
- `in_order` parameter: `None` (random), `"left_to_right"`, or `list[LongTensor]` (explicit per-sequence orders).
- Orders are padded to uniform length with position 0 (BOS) — no-op if logit formatter is correct.
- `sample_ctmc_linear_interpolation` — the `sample_ctmc_` prefix is for CTMC-based continuous-time sampling methods.

## Kortemme Tyrosine Kinase Project Reference

- Located at `/home/ishan/kortemme_tyrosine_kinase_design/`
- Uses the same EphB1 kinase domain (7KPM structure, P54762)
- Has preprocessed `7KPM_atom37_295.pt` in `structures/`
- `train_ohe_mlp.py` — comprehensive training script with OneHotMLP, EmbeddingMLP, ESMC/ESM3 probes, LoRA probe
- `data.py` — dataset loading with GuidanceDataset (old API)
- Key finding: OneHotMLP (ρ=0.46 combi activity) outperformed all ESM-based approaches for this small dataset (~780 samples)


