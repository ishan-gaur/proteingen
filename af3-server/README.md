# AF3 Inference Server

A persistent AlphaFold 3 inference server that loads the model once and serves
predictions via a FastAPI REST API. Runs inside an Apptainer container on
SLURM GPU nodes.

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────────────┐
│   Python client     │  HTTP   │  Apptainer container (GPU node)     │
│   (your script)     │────────►│  ┌─────────────────────────────────┐│
│                     │         │  │ FastAPI server (server.py)      ││
│  client.fold(seq)   │◄────────│  │  └─ ModelRunner (JAX, loaded   ││
│  client.fold_batch  │         │  │     once at startup)            ││
└─────────────────────┘         │  │  └─ InferenceWorker (job queue) ││
                                │  └─────────────────────────────────┘│
                                │  Mounts:                            │
                                │    /app/models ← /data/af3          │
                                │    /public_databases ← /data/af3db  │
                                └─────────────────────────────────────┘
```

**Why a server?** Each AF3 Docker/Apptainer invocation reloads the model (~30s)
and JIT-compiles for each input size. A persistent server loads once and reuses
the compiled model, cutting per-prediction overhead significantly.

## Quick Start

### 1. Prerequisites

- Apptainer installed (user-level is fine, see [setup](#apptainer-setup))
- AF3 Docker image built (`alphafold3:latest`)
- Model weights at `/data/af3/af3.bin`
- Genetic databases at `/data/af3db/` (optional, only if running MSA pipeline)

### 2. Build the Apptainer image (one-time)

```bash
# Convert the Docker image + add FastAPI/uvicorn
apptainer build /data/apptainer_images/alphafold3_server.sif af3_server.def
```

Or if the Docker image is already local:
```bash
apptainer build /data/apptainer_images/alphafold3_server.sif docker-daemon://alphafold3:latest
# (This won't include FastAPI — use the .def file instead)
```

### 3. Launch the server

**Via SLURM:**
```bash
sbatch launch.sh
```

**Interactively (on a GPU node):**
```bash
bash launch.sh
```

The server prints `Server ready.` once the model is loaded and it's accepting
requests. First request triggers JIT compilation for that input size (~60s),
subsequent requests of similar size are fast.

### 4. Use the client

```python
from af3_server.client import AF3Client

client = AF3Client("http://localhost:8080")

# Check server is up
client.health()  # {'status': 'ok', 'queue_size': 0}

# Fold a single protein
result = client.fold("MVLSPADKTNVKAAWGKVGAHAG...")
print(result.best_ranking_score)
print(result.summary_confidences)

# Fold a batch of proteins
results = client.fold_batch(
    sequences=["MVLS...", "GPAV...", "YKLM..."],
    names=["protein_1", "protein_2", "protein_3"],
)
for r in results:
    print(f"{r.name}: score={r.best_ranking_score:.4f}")

# Download the CIF file
client.download_cif(result.job_id, "output/my_protein.cif")
```

## API Reference

### `POST /fold`

Submit a fold job. Returns immediately with a job ID.

**Simple mode** (protein sequences):
```json
{
    "name": "my_protein",
    "sequences": ["MVLSPADKTNVKAAWG..."],
    "seeds": [0]
}
```

**Advanced mode** (raw AF3 JSON):
```json
{
    "af3_json": {
        "name": "complex",
        "modelSeeds": [1, 2],
        "sequences": [...],
        "dialect": "alphafold3",
        "version": 1
    }
}
```

### `GET /status/{job_id}`

Poll job status. Returns:
```json
{
    "job_id": "a1b2c3d4",
    "status": "completed",
    "result": {
        "name": "my_protein",
        "output_dir": "/app/af_output/a1b2c3d4",
        "ranking_scores": [{"seed": 0, "sample": 0, "ranking_score": 0.85}],
        "best_ranking_score": 0.85,
        "summary_confidences": {"ptm": 0.82, "iptm": 0.78, ...}
    }
}
```

Status values: `queued`, `running`, `completed`, `failed`.

### `GET /result/{job_id}/cif`

Download the best-ranked mmCIF structure file.

### `GET /health`

Returns `{"status": "ok", "queue_size": N}`.

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AF3_PORT` | `8080` | Server port |
| `AF3_MODEL_DIR` | `/data/af3` | Model weights directory |
| `AF3_DB_DIR` | `/data/af3db` | Genetic databases directory |
| `AF3_OUTPUT_DIR` | `/data/af3_server_output` | Prediction output directory |
| `AF3_SIF` | `/data/apptainer_images/alphafold3_server.sif` | Apptainer image path |
| `AF3_RUN_DATA_PIPELINE` | `false` | Run MSA search (slow, needs databases) |
| `AF3_GPU` | `0` | GPU index to use |
| `AF3_CACHE_DIR` | `/data/af3_jax_cache` | JAX compilation cache |

## Apptainer Setup

If Apptainer is not installed system-wide, install it at user level:

```bash
# Download and extract the deb package
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb
dpkg-deb -x apptainer_1.4.5_amd64.deb extracted/

# Copy to user directories
cp -r extracted/usr/bin/* ~/bin/
cp -r extracted/usr/libexec ~/libexec
cp -r extracted/etc/apptainer ~/etc/apptainer

# libfuse3 (needed by squashfuse)
apt download libfuse3-3
dpkg-deb -x libfuse3-3_*.deb fuse3/
mkdir -p ~/lib
cp fuse3/lib/x86_64-linux-gnu/libfuse3.so.3* ~/lib/

# Patch squashfuse_ll to find libfuse3
mv ~/libexec/apptainer/bin/squashfuse_ll ~/libexec/apptainer/bin/squashfuse_ll.real
cat > ~/libexec/apptainer/bin/squashfuse_ll << EOF
#!/bin/bash
export LD_LIBRARY_PATH="$HOME/lib:\$LD_LIBRARY_PATH"
exec "$HOME/libexec/apptainer/bin/squashfuse_ll.real" "\$@"
EOF
chmod +x ~/libexec/apptainer/bin/squashfuse_ll

# Add to PATH
echo 'export PATH=~/bin:$PATH' >> ~/.bashrc
```

## File Layout

```
af3-server/
├── README.md          ← this file
├── server.py          ← FastAPI server (runs inside container)
├── client.py          ← Python client (runs on your machine)
├── launch.sh          ← SLURM/interactive launch script
└── af3_server.def     ← Apptainer definition file
```

## Notes

- **GPU compatibility**: Tested on RTX 6000 Ada (49GB). Officially supported on
  A100/H100 80GB only, but works for typical single-chain predictions.
- **First prediction is slow** (~2-3 min) due to JIT compilation. Subsequent
  predictions of similar token count reuse the compiled model (~60-70s).
- **JAX compilation cache** persists compiled models across server restarts.
  Set `AF3_CACHE_DIR` to a persistent directory.
- **Queue**: Jobs are processed sequentially (one GPU). Multiple submissions
  are queued and processed in order.
- **No MSA by default**: The server runs with `AF3_RUN_DATA_PIPELINE=false`,
  meaning no MSA search. Designed proteins typically don't need MSA. Set to
  `true` and ensure databases are mounted if you need MSA search.
