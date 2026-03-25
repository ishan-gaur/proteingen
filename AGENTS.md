# Agent Guidelines

## Skills

Available skills:
- `add-generative-model` — workflow for integrating a new generative model into the library

## Project Management

- Use `uv` for all package management and running Python code
  - Install dependencies: `uv add <package>`
  - Run scripts: `uv run python <script>`
  - Sync environment: `uv sync`
  - Install package in editable mode: `uv pip install -e .`
  - Run tests: `uv run python -m pytest tests/ -v`
  - Run formatter: `uv run ruff format`
  - Run lineter: `uv run ruff check`
- Use this file to note down project-related info important to know across sessions
- **Discuss design decisions before implementing** - especially for abstractions and class structures
- **Ask for clarification when instructions seem contradictory** - don't guess intent, surface the confusion

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and modular
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
- `tests/` — pytest suite (current status: 222/222 passing). See → [tests/AGENTS.md](tests/AGENTS.md)
- `scripts/` — one-off experiment scripts (embedding computation, guidance experiments)
- `docs/` — MkDocs documentation site. See → [docs/AGENTS.md](docs/AGENTS.md)
- `TODO.md`, `PLAN.md`, `DESIGN.md` — roadmap and design docs


## AF3 Server

- `af3-server/` subdirectory — FastAPI server + Python client for persistent AF3 inference
- **Apptainer** installed at user level: `~/bin/apptainer` v1.4.5, `~/lib/libfuse3.so.3`, `~/libexec/apptainer/bin/squashfuse_ll` (wrapper for libfuse3)
- **SIF image**: `/data/apptainer_images/alphafold3_server.sif` (alphafold3 Docker image + FastAPI/uvicorn)
- **Model weights**: `/data/af3/af3.bin`
- **Databases**: `/data/af3db/` (original), `/data/af3db_updated/` (newer PDB from 2025/11/14)
- **RTX 6000 Ada (49GB) requires unified memory**: `TF_FORCE_UNIFIED_MEMORY=true`, `XLA_CLIENT_MEM_FRACTION=3.2` — OOMs with `XLA_PYTHON_CLIENT_PREALLOCATE=true`
- **Installed AF3 package differs from source repo**: container uses `alphafold3.model.model.Model` (not `diffusion_model.Diffuser`), `extract_inference_results` (not `extract_structures`)
- **JAX compilation cache**: `/data/af3_jax_cache` — persists compiled models across server restarts
- hanlun runs AF3 via Docker with `--norun_data_pipeline` (pre-computed MSA); their SLURM array jobs use all 4 GPUs

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `peft>=0.13.0` for LoRA adapter support
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — harmless
- **Repo renamed to `proteingen`** — display name is **ProteinGen**, URLs/paths/package-slug stay lowercase `proteingen`
- Git remote: `git@github.com:ishan-gaur/proteingen.git`


