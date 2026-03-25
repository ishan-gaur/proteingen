#!/bin/bash
# Launch the AF3 inference server inside Apptainer on a SLURM GPU node.
#
# Usage:
#   sbatch launch.sh                     # submit to SLURM
#   bash launch.sh                       # run interactively (must be on GPU node)
#
# Environment variables (override defaults):
#   AF3_PORT           — server port (default: 8080)
#   AF3_MODEL_DIR      — path to AF3 model weights (default: /data/af3)
#   AF3_DB_DIR         — path to genetic databases (default: /data/af3db)
#   AF3_OUTPUT_DIR     — where to write predictions (default: /data/af3_server_output)
#   AF3_SIF            — path to .sif image (default: /data/apptainer_images/alphafold3_server.sif)
#   AF3_RUN_DATA_PIPELINE — "true" to run MSA search (default: "false")
#   AF3_GPU            — which GPU index to use (default: 0)
#   AF3_CACHE_DIR      — JAX compilation cache dir (default: /data/af3_jax_cache)

#SBATCH --job-name=af3-server
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
AF3_PORT="${AF3_PORT:-8080}"
AF3_MODEL_DIR="${AF3_MODEL_DIR:-/data/af3}"
AF3_DB_DIR="${AF3_DB_DIR:-/data/af3db}"
AF3_OUTPUT_DIR="${AF3_OUTPUT_DIR:-/data/af3_server_output}"
AF3_SIF="${AF3_SIF:-/data/apptainer_images/alphafold3_server.sif}"
AF3_RUN_DATA_PIPELINE="${AF3_RUN_DATA_PIPELINE:-false}"
AF3_GPU="${AF3_GPU:-0}"
AF3_CACHE_DIR="${AF3_CACHE_DIR:-/data/af3_jax_cache}"

# ── Apptainer setup ──────────────────────────────────────────────────────
export PATH="$HOME/bin:$PATH"

SERVER_SCRIPT="$(cd "$(dirname "$0")" && pwd)/server.py"

echo "================================================================"
echo "AF3 Inference Server"
echo "================================================================"
echo "SIF image:      $AF3_SIF"
echo "Server script:  $SERVER_SCRIPT"
echo "Model dir:      $AF3_MODEL_DIR"
echo "DB dir:         $AF3_DB_DIR"
echo "Output dir:     $AF3_OUTPUT_DIR"
echo "Port:           $AF3_PORT"
echo "GPU:            $AF3_GPU"
echo "Data pipeline:  $AF3_RUN_DATA_PIPELINE"
echo "JAX cache:      $AF3_CACHE_DIR"
echo "================================================================"

mkdir -p "$AF3_OUTPUT_DIR" "$AF3_CACHE_DIR"

# Bind-mount paths:
#   /data/af3         → /app/models      (model weights)
#   /data/af3db       → /public_databases (genetic databases)
#   AF3_OUTPUT_DIR    → /app/af_output    (predictions)
#   server.py dir     → /app/server       (our server code)
#   AF3_CACHE_DIR     → /app/jax_cache    (compilation cache)

export CUDA_VISIBLE_DEVICES="$AF3_GPU"

apptainer exec \
    --nv \
    --bind "$AF3_MODEL_DIR:/app/models" \
    --bind "$AF3_DB_DIR:/public_databases" \
    --bind "$AF3_OUTPUT_DIR:/app/af_output" \
    --bind "$(dirname "$SERVER_SCRIPT"):/app/server" \
    --bind "$AF3_CACHE_DIR:/app/jax_cache" \
    --env AF3_MODEL_DIR=/app/models \
    --env AF3_DB_DIR=/public_databases \
    --env AF3_OUTPUT_DIR=/app/af_output \
    --env AF3_PORT="$AF3_PORT" \
    --env AF3_RUN_DATA_PIPELINE="$AF3_RUN_DATA_PIPELINE" \
    --env XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env TF_FORCE_UNIFIED_MEMORY=true \
    --env XLA_CLIENT_MEM_FRACTION=3.2 \
    --env JAX_COMPILATION_CACHE_DIR=/app/jax_cache \
    --pwd /app/alphafold \
    "$AF3_SIF" \
    python3 /app/server/server.py
