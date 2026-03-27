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
                                │    /app/models ← model weights      │
                                │    /public_databases ← genetic DBs  │
                                └─────────────────────────────────────┘
```

**Why a server?** Each AF3 Docker/Apptainer invocation reloads model weights
(~30s) and JIT-compiles for each input size. A persistent server loads once and
reuses the compiled model, cutting per-prediction overhead significantly.


## Setup From Scratch

This section walks through everything needed to get the server running on a new
machine. Tested on Ubuntu 20.04 with NVIDIA RTX 6000 Ada GPUs and SLURM.

### Step 1: Get AlphaFold 3

Clone the AF3 source code and build the Docker image:

```bash
git clone https://github.com/google-deepmind/alphafold3.git
cd alphafold3
docker build -f docker/Dockerfile -t alphafold3 .
```

This builds a Docker image with all AF3 dependencies (JAX, HMMER, etc.).

### Step 2: Obtain Model Weights

Request access at https://forms.gle/svvpY4u2jsHEwWYS6 (Google DeepMind will
email you). Download the model parameters and place them in a directory:

```bash
mkdir -p /data/af3
# Place af3.bin in this directory
ls /data/af3/af3.bin
```

### Step 3: Obtain Genetic Databases (Optional)

Only needed if you want to run MSA search (`AF3_RUN_DATA_PIPELINE=true`).
For folding designed proteins, you typically don't need this.

```bash
# The AF3 repo includes a script to download databases (~1TB)
bash fetch_databases.sh /data/af3db
```

This downloads UniRef90, BFD, MGnify, PDB, Rfam, RNAcentral, etc.

### Step 4: Install Apptainer

SLURM clusters typically don't allow Docker (requires root). Apptainer
(formerly Singularity) runs containers without privileges.

**If you have sudo access:**
```bash
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

**If you don't have sudo (user-level install):**

```bash
# 1. Download and extract the Apptainer deb package
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb
dpkg-deb -x apptainer_1.4.5_amd64.deb extracted/

# 2. Copy binaries and config to your home directory
mkdir -p ~/bin ~/lib
cp -r extracted/usr/bin/* ~/bin/
cp -r extracted/usr/libexec ~/libexec
cp -r extracted/etc/apptainer ~/etc/apptainer

# 3. Install libfuse3 (required by Apptainer's squashfuse)
apt download libfuse3-3
dpkg-deb -x libfuse3-3_*.deb fuse3/
cp fuse3/lib/x86_64-linux-gnu/libfuse3.so.3.* ~/lib/libfuse3.so.3

# 4. Patch squashfuse_ll to find libfuse3
#    Apptainer launches squashfuse_ll as a subprocess that doesn't inherit
#    LD_LIBRARY_PATH, so we wrap it with a script that sets the path.
mv ~/libexec/apptainer/bin/squashfuse_ll ~/libexec/apptainer/bin/squashfuse_ll.real
cat > ~/libexec/apptainer/bin/squashfuse_ll << 'WRAPPER'
#!/bin/bash
export LD_LIBRARY_PATH="HOMEDIR/lib:$LD_LIBRARY_PATH"
exec "HOMEDIR/libexec/apptainer/bin/squashfuse_ll.real" "$@"
WRAPPER
# Replace HOMEDIR with your actual home directory
sed -i "s|HOMEDIR|$HOME|g" ~/libexec/apptainer/bin/squashfuse_ll
chmod +x ~/libexec/apptainer/bin/squashfuse_ll

# 5. Add ~/bin to PATH
echo 'export PATH=~/bin:$PATH' >> ~/.bashrc
export PATH=~/bin:$PATH

# 6. Verify
apptainer --version  # should print "apptainer version 1.4.5"
apptainer exec docker://alpine:latest cat /etc/os-release  # quick test
```

**Requirements for user-level install:**
- Linux kernel with user namespaces enabled (check: `cat /proc/sys/user/max_user_namespaces` should be > 0)
- `wget`, `dpkg-deb` available

### Step 5: Build the Server Container Image

This creates an Apptainer SIF image from the AF3 Docker image with FastAPI
and uvicorn added:

```bash
# From this directory (af3-server/)
apptainer build /data/apptainer_images/alphafold3_server.sif af3_server.def
```

This takes ~2-5 minutes. The resulting `.sif` file is ~7GB.

If you'd rather convert the Docker image directly (without FastAPI baked in),
you can, but then you'll need to install FastAPI at runtime via an overlay or
`--writable-tmpfs` — the `.def` file approach is simpler.

### Step 6: Launch the Server

Edit the paths in `launch.sh` if your data is in a different location, then:

**Via SLURM (recommended):**
```bash
sbatch launch.sh
```

**Interactively (must be on a GPU node):**
```bash
bash launch.sh
```

Watch the log for `Server ready.`:
```bash
tail -f ~/slurm/output/af3-server.*.out  # if submitted via SLURM
```

### Step 7: Use It

From any machine that can reach the server (same node, or another node on the
same network):

```python
from client import AF3Client

client = AF3Client("http://localhost:8080")
client.health()  # {'status': 'ok', 'queue_size': 0}

result = client.fold("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")
print(f"Score: {result.best_ranking_score:.3f}")  # ~0.83 for ubiquitin
```

See [Client API](#client-api) below for full usage.


## Client API

### `AF3Client(base_url, poll_interval=5.0, timeout=3600.0)`

```python
from client import AF3Client
client = AF3Client("http://localhost:8080")
```

### `client.fold(sequence, name="fold_job", seeds=None) → FoldResult`

Fold a single protein. Blocks until complete.

```python
result = client.fold("MVLSPADKTNVKAAWGKVGAHAG...")
print(result.best_ranking_score)   # float, e.g. 0.83
print(result.summary_confidences)  # dict with ptm, iptm, etc.
print(result.output_dir)           # server-side path to all outputs
```

### `client.fold_batch(sequences, names=None, seeds=None) → list[FoldResult]`

Fold multiple proteins, each as a separate single-chain job. Submits all at
once; the server queues them and processes sequentially.

```python
results = client.fold_batch(
    ["MVLS...", "GPAV...", "YKLM..."],
    names=["design_1", "design_2", "design_3"],
)
for r in results:
    print(f"{r.name}: {r.best_ranking_score:.3f}")
```

### `client.fold_complex(sequences, name="complex", seeds=None) → FoldResult`

Fold a multi-chain complex. Each sequence becomes a chain (A, B, C, ...).

```python
result = client.fold_complex(["MVLS...", "GPAV..."], name="dimer")
```

### `client.submit(sequences, name, seeds) → str`

Non-blocking: submit a job and get back the job ID.

### `client.wait(job_id) → FoldResult`

Block until a submitted job completes.

### `client.status(job_id) → dict`

Poll job status without blocking. Status is one of:
`"queued"`, `"running"`, `"completed"`, `"failed"`.

### `client.download_cif(job_id, output_path) → Path`

Download the best-ranked mmCIF structure file to a local path.

```python
client.download_cif(result.job_id, "structures/my_protein.cif")
```

### `FoldResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | `str` | Server job ID |
| `name` | `str` | Job name |
| `best_ranking_score` | `float` | Highest ranking score across samples |
| `ranking_scores` | `list[dict]` | Per-seed, per-sample scores |
| `summary_confidences` | `dict` | pTM, ipTM, pLDDT, disorder, clashes |
| `output_dir` | `str` | Server-side output directory |
| `elapsed_seconds` | `float` | Wall-clock time for the job |


## REST API Reference

### `POST /fold`

Submit a fold job. Returns immediately with a job ID.

**Simple mode** — list of protein sequences:
```json
{
    "name": "my_protein",
    "sequences": ["MVLSPADKTNVKAAWG..."],
    "seeds": [0]
}
```

**Advanced mode** — raw AF3 JSON (see [AF3 input docs](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)):
```json
{
    "af3_json": {
        "name": "complex",
        "modelSeeds": [1, 2],
        "sequences": [
            {"protein": {"id": "A", "sequence": "MVLS..."}},
            {"ligand": {"id": "B", "ccdCodes": ["ATP"]}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
}
```

### `GET /status/{job_id}`

Returns:
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

### `GET /result/{job_id}/cif`

Download the best-ranked mmCIF file. Returns `chemical/x-mmcif` content.

### `GET /health`

Returns `{"status": "ok", "queue_size": N}`.

### Interactive docs

FastAPI auto-generates interactive API documentation at
`http://localhost:8080/docs` (Swagger UI).


## Configuration

All via environment variables (set in `launch.sh` or override on command line):

| Variable | Default | Description |
|----------|---------|-------------|
| `AF3_PORT` | `8080` | Server port |
| `AF3_MODEL_DIR` | `/data/af3` | Directory containing `af3.bin` |
| `AF3_DB_DIR` | `/data/af3db` | Genetic databases (for MSA pipeline) |
| `AF3_OUTPUT_DIR` | `/data/af3_server_output` | Where predictions are written |
| `AF3_SIF` | `/data/apptainer_images/alphafold3_server.sif` | Apptainer image |
| `AF3_RUN_DATA_PIPELINE` | `false` | Run MSA search (needs databases + time) |
| `AF3_GPU` | `0` | GPU index |
| `AF3_CACHE_DIR` | `/data/af3_jax_cache` | JAX compilation cache |


## File Layout

```
af3-server/
├── README.md          ← this file
├── server.py          ← FastAPI server (runs inside container)
├── client.py          ← Python client (runs outside container)
├── launch.sh          ← SLURM / interactive launch script
├── test_server.sh     ← End-to-end SLURM test script
├── example.py         ← Demo: fold from CLI or FASTA
└── af3_server.def     ← Apptainer definition file
```


## GPU Memory & Compatibility

AF3 is officially supported on **A100 80GB** and **H100 80GB**. It works on
smaller GPUs with adjusted memory settings.

**For GPUs < 80GB** (e.g. RTX 6000 Ada 49GB, A100 40GB), unified memory is
required. This is already configured in `launch.sh`:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
TF_FORCE_UNIFIED_MEMORY=true
XLA_CLIENT_MEM_FRACTION=3.2
```
This allows JAX to spill GPU memory to host RAM when needed. Inference is
slightly slower but works correctly.

**For A100/H100 80GB**, you can use the default preallocated mode for better
performance:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_CLIENT_MEM_FRACTION=0.95
```

### Tested Performance (RTX 6000 Ada, 49GB)

| Input | Tokens | Inference time | Ranking score |
|-------|--------|---------------|---------------|
| Ubiquitin (76 residues) | ~256 | 44s | 0.83 |

First prediction for a given token bucket includes JAX JIT compilation. The
**JAX compilation cache** (`AF3_CACHE_DIR`) persists compiled models across
server restarts so you don't pay this cost again.


## Troubleshooting

### Server won't start: "No GPU devices found"
Make sure you're running on a GPU node. In SLURM, the `--gres=gpu:1` flag
in `launch.sh` handles this. For interactive use, ensure `CUDA_VISIBLE_DEVICES`
is set and `nvidia-smi` shows a GPU.

### OOM during inference
Switch to unified memory (see [GPU Memory](#gpu-memory--compatibility)). If
you're already using unified memory, ensure the host has enough RAM (64GB+
recommended).

### "squashfuse_ll: error while loading shared libraries: libfuse3.so.3"
The `squashfuse_ll` wrapper script isn't finding libfuse3. Check that:
1. `~/lib/libfuse3.so.3` exists
2. The wrapper at `~/libexec/apptainer/bin/squashfuse_ll` has the correct
   absolute path to your home directory (not `$HOME` — it must be expanded)

### "can't open file '/app/server/server.py'"
The bind mount for the server script isn't working. In SLURM, use absolute paths
(not `$(dirname "$0")` which can break when SLURM copies the script). Check the
`--bind` flags in `launch.sh`.

### Job stuck in "running" forever
AF3 inference for large proteins (>1000 tokens) can take 5-40 minutes. The
client's `timeout` parameter (default 3600s) controls how long `wait()` blocks.
Check server logs for progress.

### "RESOURCE_EXHAUSTED" on A100/H100
You may be hitting the 5120-token limit. For larger inputs, add larger bucket
sizes or use unified memory.

### Multiple users sharing GPUs
The server uses one GPU exclusively. On a shared cluster, use SLURM to allocate
GPUs properly. If other users run Docker containers outside SLURM, their jobs
can consume GPU memory that SLURM thinks is free — coordinate with other users
or use `hold.sh`-style reservation scripts to claim GPUs first.
