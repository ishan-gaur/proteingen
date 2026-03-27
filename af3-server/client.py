"""Client for the AF3 inference server.

Usage:
    from af3_server.client import AF3Client

    client = AF3Client("http://localhost:8080")

    # Fold a single protein
    result = client.fold("MVLSPADKTNVKAAWGKVGAHAG...")
    print(result.ranking_score, result.cif_path)

    # Fold multiple proteins (queued sequentially on the server)
    results = client.fold_batch(["MVLS...", "GPAV...", "YKLM..."])
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import requests


@dataclasses.dataclass(frozen=True)
class FoldResult:
    """Result from a single fold job."""

    job_id: str
    name: str
    ranking_scores: list[dict[str, Any]]
    best_ranking_score: float
    summary_confidences: dict[str, Any]
    output_dir: str
    elapsed_seconds: float

    @property
    def plddt(self) -> float | None:
        """Global pLDDT if available in summary confidences."""
        return self.summary_confidences.get("ptm")

    @property
    def iptm(self) -> float | None:
        """Interface pTM if available."""
        return self.summary_confidences.get("iptm")


class AF3Client:
    """Client for the AF3 inference server.

    Args:
        base_url: Server URL, e.g. "http://localhost:8080"
        poll_interval: Seconds between status polls (default: 5)
        timeout: Max seconds to wait for a job to complete (default: 3600)
    """

    def __init__(
        self,
        base_url: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout

    def health(self) -> dict:
        """Check server health."""
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def submit(
        self,
        sequences: list[str],
        name: str = "fold_job",
        seeds: list[int] | None = None,
    ) -> str:
        """Submit a fold job and return the job_id (non-blocking)."""
        payload: dict[str, Any] = {"name": name, "sequences": sequences}
        if seeds is not None:
            payload["seeds"] = seeds
        r = requests.post(f"{self.base_url}/fold", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["job_id"]

    def submit_json(self, af3_json: dict) -> str:
        """Submit a raw AF3 JSON input and return the job_id."""
        payload = {"af3_json": af3_json}
        r = requests.post(f"{self.base_url}/fold", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["job_id"]

    def status(self, job_id: str) -> dict:
        """Get job status."""
        r = requests.get(f"{self.base_url}/status/{job_id}", timeout=10)
        r.raise_for_status()
        return r.json()

    def wait(self, job_id: str) -> FoldResult:
        """Block until a job completes and return the result."""
        t0 = time.time()
        while True:
            s = self.status(job_id)
            if s["status"] == "completed":
                return FoldResult(
                    job_id=job_id,
                    name=s["result"]["name"],
                    ranking_scores=s["result"]["ranking_scores"],
                    best_ranking_score=s["result"]["best_ranking_score"],
                    summary_confidences=s["result"]["summary_confidences"],
                    output_dir=s["result"]["output_dir"],
                    elapsed_seconds=time.time() - t0,
                )
            if s["status"] == "failed":
                raise RuntimeError(f"Job {job_id} failed: {s.get('error')}")
            if time.time() - t0 > self.timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {self.timeout}s "
                    f"(status: {s['status']})"
                )
            time.sleep(self.poll_interval)

    def download_cif(self, job_id: str, output_path: str | pathlib.Path) -> pathlib.Path:
        """Download the best-ranked CIF file for a completed job."""
        output_path = pathlib.Path(output_path)
        r = requests.get(f"{self.base_url}/result/{job_id}/cif", timeout=30)
        r.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(r.content)
        return output_path

    def fold(
        self,
        sequence: str,
        name: str = "fold_job",
        seeds: list[int] | None = None,
    ) -> FoldResult:
        """Fold a single protein sequence (blocking)."""
        job_id = self.submit([sequence], name=name, seeds=seeds)
        return self.wait(job_id)

    def fold_batch(
        self,
        sequences: list[str],
        names: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> list[FoldResult]:
        """Fold multiple proteins, each as a separate single-chain job.

        Jobs are submitted all at once and processed sequentially by the server.
        This method blocks until all jobs complete.
        """
        if names is None:
            names = [f"seq_{i}" for i in range(len(sequences))]
        assert len(names) == len(sequences)

        # Submit all jobs
        job_ids = []
        for seq, name in zip(sequences, names):
            job_id = self.submit([seq], name=name, seeds=seeds)
            job_ids.append(job_id)
            print(f"Submitted {name} → {job_id}")

        # Wait for all
        results = []
        for job_id, name in zip(job_ids, names):
            print(f"Waiting for {name} ({job_id})...", end=" ", flush=True)
            result = self.wait(job_id)
            print(f"done ({result.elapsed_seconds:.1f}s, score={result.best_ranking_score:.4f})")
            results.append(result)

        return results

    def fold_complex(
        self,
        sequences: list[str],
        name: str = "complex",
        seeds: list[int] | None = None,
    ) -> FoldResult:
        """Fold a multi-chain complex (blocking)."""
        job_id = self.submit(sequences, name=name, seeds=seeds)
        return self.wait(job_id)
