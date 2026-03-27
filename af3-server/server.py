"""AlphaFold 3 inference server.

Loads the AF3 model once at startup, then serves predictions via a FastAPI REST
API. Designed to run inside the AF3 Apptainer container with GPUs available.

Endpoints:
    POST /fold        — submit a fold job (returns job_id immediately)
    GET  /status/{id} — poll job status / retrieve results
    GET  /result/{id}/cif — download best-ranked CIF
    GET  /health      — liveness check
"""

import dataclasses
import functools
import json
import os
import pathlib
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable

import fastapi
import fastapi.responses
import pydantic
import uvicorn

# ---------------------------------------------------------------------------
# AF3 imports (available inside the container)
# ---------------------------------------------------------------------------
from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Pydantic models for the API
# ---------------------------------------------------------------------------

CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

BUCKETS = (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)


class FoldRequest(pydantic.BaseModel):
    """Request body for POST /fold.

    Two modes:
    1. Simple: provide `sequences` (list of protein sequence strings).
    2. Advanced: provide `af3_json` (raw AF3 input JSON dict).
    """

    name: str = "fold_job"
    sequences: list[str] | None = None
    seeds: list[int] = [0]
    num_diffusion_samples: int = 5
    num_recycles: int = 10
    af3_json: dict | None = None

    @pydantic.model_validator(mode="after")
    def check_input(self) -> "FoldRequest":
        if self.sequences is None and self.af3_json is None:
            raise ValueError("Must provide either 'sequences' or 'af3_json'")
        return self


class JobStatus(pydantic.BaseModel):
    job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    error: str | None = None
    result: dict | None = None
    submitted_at: float
    started_at: float | None = None
    completed_at: float | None = None


# ---------------------------------------------------------------------------
# ModelRunner — adapted from the container's run_alphafold.py
# ---------------------------------------------------------------------------


def make_model_config(
    flash_attention_implementation: attention.Implementation = "triton",
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
) -> model.Model.Config:
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = (
        flash_attention_implementation
    )
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = False
    config.return_distogram = False
    return config


class ModelRunner:
    """Loads AF3 model once and runs inference."""

    def __init__(
        self,
        config: model.Model.Config,
        device: jax.Device,
        model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        print("Loading model parameters...", flush=True)
        p = params.get_model_haiku_params(model_dir=self._model_dir)
        print("Model parameters loaded.", flush=True)
        return p

    @functools.cached_property
    def _model(
        self,
    ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> model.ModelResult:
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray,
                utils.remove_invalidly_typed_feats(featurised_example),
            ),
            self._device,
        )
        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params["__meta__"]["__identifier__"].tobytes()
        result["__identifier__"] = identifier
        return result

    def extract_inference_results(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )


# ---------------------------------------------------------------------------
# Job queue and worker
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Job:
    job_id: str
    fold_input: folding_input.Input
    status: JobStatus
    output_dir: pathlib.Path


class InferenceWorker:
    """Background thread that processes fold jobs sequentially on the GPU."""

    def __init__(
        self,
        model_runner: ModelRunner,
        data_pipeline_config: pipeline.DataPipelineConfig | None,
        output_base_dir: pathlib.Path,
    ):
        self.model_runner = model_runner
        self.data_pipeline_config = data_pipeline_config
        self.output_base_dir = output_base_dir
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, fold_input: folding_input.Input) -> str:
        job_id = str(uuid.uuid4())[:8]
        output_dir = self.output_base_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        status = JobStatus(
            job_id=job_id,
            status="queued",
            submitted_at=time.time(),
        )
        job = Job(
            job_id=job_id,
            fold_input=fold_input,
            status=status,
            output_dir=output_dir,
        )
        self._jobs[job_id] = job
        self._queue.put(job)
        return job_id

    def get_status(self, job_id: str) -> JobStatus | None:
        job = self._jobs.get(job_id)
        return job.status if job else None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            job.status.status = "running"
            job.status.started_at = time.time()
            print(
                f"[{job.job_id}] Starting inference for '{job.fold_input.name}'",
                flush=True,
            )

            try:
                fi = job.fold_input

                # Run data pipeline (MSA search) if configured
                if self.data_pipeline_config is not None:
                    print(f"[{job.job_id}] Running data pipeline...", flush=True)
                    fi = pipeline.DataPipeline(self.data_pipeline_config).process(fi)

                # Featurise
                print(f"[{job.job_id}] Featurising...", flush=True)
                ccd = chemical_components.cached_ccd(user_ccd=fi.user_ccd)
                featurised_examples = featurisation.featurise_input(
                    fold_input=fi, buckets=BUCKETS, ccd=ccd, verbose=False
                )

                # Run inference for each seed
                all_inference_results = []
                for seed, example in zip(fi.rng_seeds, featurised_examples):
                    print(f"[{job.job_id}] Inference seed={seed}...", flush=True)
                    t0 = time.time()
                    rng_key = jax.random.PRNGKey(seed)
                    result = self.model_runner.run_inference(example, rng_key)
                    inference_results = (
                        self.model_runner.extract_inference_results(
                            batch=example, result=result, target_name=fi.name
                        )
                    )
                    dt = time.time() - t0
                    print(
                        f"[{job.job_id}] Seed {seed} took {dt:.1f}s", flush=True
                    )
                    all_inference_results.extend(
                        (seed, idx, ir)
                        for idx, ir in enumerate(inference_results)
                    )

                # Write outputs
                ranking_scores = []
                best_score = None
                best_result = None

                for seed, sample_idx, ir in all_inference_results:
                    sample_dir = (
                        job.output_dir / f"seed-{seed}_sample-{sample_idx}"
                    )
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    post_processing.write_output(
                        inference_result=ir, output_dir=str(sample_dir)
                    )
                    score = float(ir.metadata["ranking_score"])
                    ranking_scores.append(
                        {
                            "seed": seed,
                            "sample": sample_idx,
                            "ranking_score": score,
                        }
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_result = ir

                # Write best result at top level
                if best_result is not None:
                    post_processing.write_output(
                        inference_result=best_result,
                        output_dir=str(job.output_dir),
                        name=fi.sanitised_name(),
                    )

                # Read summary confidences
                summary_path = (
                    job.output_dir
                    / f"{fi.sanitised_name()}_summary_confidences.json"
                )
                summary_confidences = {}
                if summary_path.exists():
                    summary_confidences = json.loads(summary_path.read_text())

                job.status.status = "completed"
                job.status.completed_at = time.time()
                job.status.result = {
                    "name": fi.name,
                    "output_dir": str(job.output_dir),
                    "ranking_scores": ranking_scores,
                    "best_ranking_score": best_score,
                    "summary_confidences": summary_confidences,
                }
                dt = job.status.completed_at - job.status.started_at
                print(
                    f"[{job.job_id}] Completed in {dt:.1f}s, "
                    f"best score={best_score:.4f}",
                    flush=True,
                )

            except Exception as e:
                job.status.status = "failed"
                job.status.completed_at = time.time()
                job.status.error = str(e)
                print(f"[{job.job_id}] FAILED: {e}", flush=True)
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Build AF3 FoldInput from simple sequence list
# ---------------------------------------------------------------------------


def sequences_to_fold_input(
    sequences: list[str],
    name: str = "fold_job",
    seeds: list[int] | None = None,
) -> folding_input.Input:
    """Convert a list of protein sequences to an AF3 FoldInput.

    Sequences are assigned chain IDs A, B, C, ... in order.
    MSA fields are set to empty strings (no MSA search needed).
    """
    if seeds is None:
        seeds = [0]
    assert len(sequences) <= len(CHAIN_IDS), f"Max {len(CHAIN_IDS)} chains"
    assert all(sequences), "Empty sequence not allowed"

    chains = []
    for i, seq in enumerate(sequences):
        chains.append(
            folding_input.ProteinChain(
                id=CHAIN_IDS[i],
                sequence=seq,
                ptms=(),
                paired_msa="",
                unpaired_msa="",
                templates=(),
            )
        )

    return folding_input.Input(
        name=name,
        chains=chains,
        rng_seeds=seeds,
        bonded_atom_pairs=None,
        user_ccd=None,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = fastapi.FastAPI(
    title="AlphaFold 3 Inference Server",
    description="Persistent AF3 server for protein structure prediction.",
    version="0.1.0",
)

_worker: InferenceWorker | None = None


@app.on_event("startup")
def startup():
    global _worker

    model_dir = pathlib.Path(os.environ.get("AF3_MODEL_DIR", "/app/models"))
    output_dir = pathlib.Path(os.environ.get("AF3_OUTPUT_DIR", "/app/af_output"))
    db_dir = os.environ.get("AF3_DB_DIR", "")
    run_data_pipeline = (
        os.environ.get("AF3_RUN_DATA_PIPELINE", "false").lower() == "true"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir: {model_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"DB dir: {db_dir}", flush=True)
    print(f"Run data pipeline: {run_data_pipeline}", flush=True)

    # Initialize GPU
    devices = jax.local_devices(backend="gpu")
    print(f"GPU devices: {devices}", flush=True)
    assert len(devices) > 0, "No GPU devices found"

    config = make_model_config()
    model_runner = ModelRunner(
        config=config, device=devices[0], model_dir=model_dir
    )

    # Eagerly load params so first request isn't slow
    _ = model_runner.model_params

    # Data pipeline config
    data_pipeline_config = None
    if run_data_pipeline and db_dir:
        db_path = pathlib.Path(db_dir)
        data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path="jackhmmer",
            nhmmer_binary_path="nhmmer",
            hmmalign_binary_path="hmmalign",
            hmmsearch_binary_path="hmmsearch",
            hmmbuild_binary_path="hmmbuild",
            small_bfd_database_path=str(
                db_path / "bfd-first_non_consensus_sequences.fasta"
            ),
            mgnify_database_path=str(db_path / "mgy_clusters_2022_05.fa"),
            uniprot_cluster_annot_database_path=str(
                db_path / "uniprot_all_2021_04.fa"
            ),
            uniref90_database_path=str(db_path / "uniref90_2022_05.fa"),
            ntrna_database_path=str(
                db_path
                / "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
            ),
            rfam_database_path=str(
                db_path
                / "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
            ),
            rna_central_database_path=str(
                db_path
                / "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
            ),
            pdb_database_path=str(db_path / "mmcif_files"),
            seqres_database_path=str(
                db_path / "pdb_seqres_2022_09_28.fasta"
            ),
        )

    _worker = InferenceWorker(
        model_runner=model_runner,
        data_pipeline_config=data_pipeline_config,
        output_base_dir=output_dir,
    )
    print("Server ready.", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "queue_size": _worker.queue_size if _worker else 0,
    }


@app.post("/fold", response_model=JobStatus)
def fold(request: FoldRequest):
    assert _worker is not None, "Server not initialized"

    if request.af3_json is not None:
        fold_input = folding_input.Input.from_json(json.dumps(request.af3_json))
    else:
        fold_input = sequences_to_fold_input(
            sequences=request.sequences,
            name=request.name,
            seeds=request.seeds,
        )

    job_id = _worker.submit(fold_input)
    return _worker.get_status(job_id)


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    assert _worker is not None, "Server not initialized"
    s = _worker.get_status(job_id)
    if s is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Job {job_id} not found"
        )
    return s


@app.get("/result/{job_id}/cif")
def get_cif(job_id: str):
    """Download the best-ranked CIF file for a completed job."""
    assert _worker is not None
    s = _worker.get_status(job_id)
    if s is None:
        raise fastapi.HTTPException(
            status_code=404, detail=f"Job {job_id} not found"
        )
    if s.status != "completed":
        raise fastapi.HTTPException(
            status_code=400, detail=f"Job {s.status}, not completed"
        )

    output_dir = pathlib.Path(s.result["output_dir"])
    # Best-ranked CIF is at the top level (not in seed-X_sample-Y subdirs)
    top_cifs = [
        f for f in output_dir.glob("*_model.cif") if f.parent == output_dir
    ]
    if not top_cifs:
        raise fastapi.HTTPException(
            status_code=404, detail="No CIF file found"
        )

    return fastapi.responses.FileResponse(
        top_cifs[0],
        media_type="chemical/x-mmcif",
        filename=top_cifs[0].name,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("AF3_PORT", "8080"))
    host = os.environ.get("AF3_HOST", "0.0.0.0")
    print(f"Starting AF3 server on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
