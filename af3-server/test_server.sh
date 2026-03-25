#!/bin/bash
#SBATCH --cluster=kraken
#SBATCH --partition=long
#SBATCH --account=researchers
#SBATCH --job-name=af3-server-test
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --qos=high-prio
#SBATCH --output=/home/ishan/slurm/output/af3_server_test.out
#SBATCH --error=/home/ishan/slurm/output/af3_server_test.err

# Test the AF3 inference server end-to-end:
#   1. Start the server inside Apptainer
#   2. Wait for it to be ready
#   3. Submit a fold request via the client
#   4. Verify the output
#   5. Shut down

set -euo pipefail

echo "=========================================="
echo "AF3 Server End-to-End Test"
echo "=========================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

export PATH="$HOME/bin:$PATH"

# Use absolute path — SLURM may copy the script, making $(dirname $0) unreliable
SCRIPT_DIR="/home/ishan/dfm-worktrees/esm3-ephb1-finetuning-worktrees/af3/af3-server"
AF3_PORT=8080
AF3_MODEL_DIR=/data/af3
AF3_DB_DIR=/data/af3db
AF3_OUTPUT_DIR=/tmp/af3_server_test_$$
AF3_SIF=/data/apptainer_images/alphafold3_server.sif
AF3_CACHE_DIR=/data/af3_jax_cache

mkdir -p "$AF3_OUTPUT_DIR" "$AF3_CACHE_DIR"

echo ""
echo "[1/4] Starting AF3 server..."
apptainer exec \
    --nv \
    --bind "$AF3_MODEL_DIR:/app/models" \
    --bind "$AF3_DB_DIR:/public_databases" \
    --bind "$AF3_OUTPUT_DIR:/app/af_output" \
    --bind "$SCRIPT_DIR:/app/server" \
    --bind "$AF3_CACHE_DIR:/app/jax_cache" \
    --env AF3_MODEL_DIR=/app/models \
    --env AF3_DB_DIR=/public_databases \
    --env AF3_OUTPUT_DIR=/app/af_output \
    --env AF3_PORT="$AF3_PORT" \
    --env AF3_RUN_DATA_PIPELINE=false \
    --env XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env TF_FORCE_UNIFIED_MEMORY=true \
    --env XLA_CLIENT_MEM_FRACTION=3.2 \
    --env JAX_COMPILATION_CACHE_DIR=/app/jax_cache \
    --pwd /app/alphafold \
    "$AF3_SIF" \
    python3 /app/server/server.py &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo ""
echo "[2/4] Waiting for server to be ready..."
MAX_WAIT=600  # 10 minutes for model loading + JIT
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$AF3_PORT/health" > /dev/null 2>&1; then
        echo "Server ready after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  Still waiting... (${WAITED}s)"
    fi
    # Check server hasn't crashed
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died!"
        exit 1
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not start within ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Health check
echo ""
curl -s "http://localhost:$AF3_PORT/health" | python3 -m json.tool
echo ""

# Submit a test fold (short protein — ubiquitin, 76 residues)
echo "[3/4] Submitting test fold job (ubiquitin, 76 residues)..."
RESPONSE=$(curl -s -X POST "http://localhost:$AF3_PORT/fold" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "ubiquitin_test",
        "sequences": ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"],
        "seeds": [0]
    }')
echo "Submit response:"
echo "$RESPONSE" | python3 -m json.tool

JOB_ID=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"

# Poll until complete
echo ""
echo "Polling for completion..."
MAX_POLL=600  # 10 min max for inference
POLLED=0
while [ $POLLED -lt $MAX_POLL ]; do
    STATUS=$(curl -s "http://localhost:$AF3_PORT/status/$JOB_ID")
    JOB_STATUS=$(echo "$STATUS" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
    
    if [ "$JOB_STATUS" = "completed" ]; then
        echo "Job completed after ${POLLED}s!"
        echo "$STATUS" | python3 -m json.tool
        break
    elif [ "$JOB_STATUS" = "failed" ]; then
        echo "Job FAILED!"
        echo "$STATUS" | python3 -m json.tool
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    
    sleep 10
    POLLED=$((POLLED + 10))
    if [ $((POLLED % 30)) -eq 0 ]; then
        echo "  Status: $JOB_STATUS (${POLLED}s)"
    fi
done

if [ $POLLED -ge $MAX_POLL ]; then
    echo "ERROR: Job did not complete within ${MAX_POLL}s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Download CIF
echo ""
echo "[4/4] Downloading CIF file..."
CIF_PATH="$AF3_OUTPUT_DIR/ubiquitin_test.cif"
curl -s "http://localhost:$AF3_PORT/result/$JOB_ID/cif" -o "$CIF_PATH"
if [ -f "$CIF_PATH" ] && [ -s "$CIF_PATH" ]; then
    echo "CIF saved to $CIF_PATH ($(wc -c < "$CIF_PATH") bytes)"
    head -5 "$CIF_PATH"
else
    echo "ERROR: CIF file not saved or empty"
fi

# List output dir
echo ""
echo "Output directory:"
find "$AF3_OUTPUT_DIR" -type f | head -20

echo ""
echo "=========================================="
echo "TEST PASSED"
echo "=========================================="

# Shut down server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo "Server shut down."
