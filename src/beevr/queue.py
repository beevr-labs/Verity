"""Ingestion job queue — doc 22 §1, FR-KM-14.

Postgres-native pattern (SELECT ... FOR UPDATE SKIP LOCKED) modelled in memory.
Stage chain per document: parse -> ocr -> chunk -> embed -> extract_graph.
Each stage: retry with exponential backoff; crash-recovery via lease reclaim
(no duplicate output); dead-letter after max_attempts with `failed` surfaced.

Time is passed in explicitly (deterministic, testable); production uses now().
"""
from __future__ import annotations

from dataclasses import dataclass, field

STAGES = ("parse", "ocr", "chunk", "embed", "extract_graph")


def _next_stage(stage: str) -> str | None:
    i = STAGES.index(stage)
    return STAGES[i + 1] if i + 1 < len(STAGES) else None


@dataclass
class Job:
    id: str
    document_id: str
    stage: str
    state: str = "queued"            # queued|running|done|failed|dead
    attempts: int = 0
    max_attempts: int = 5
    run_after: float = 0.0
    locked_by: str | None = None
    locked_at: float | None = None
    last_error: str | None = None


class IngestQueue:
    def __init__(self, base_backoff: float = 5.0, lease_timeout: float = 900.0):
        self.jobs: dict[str, Job] = {}
        self.completed: dict[str, set[str]] = {}
        self.dead: dict[str, str] = {}
        self.output: dict[tuple[str, str], object] = {}  # (doc,stage) -> result (upsert)
        self.base_backoff = base_backoff
        self.lease_timeout = lease_timeout
        self._seq = 0

    def _new_job(self, document_id: str, stage: str, now: float,
                 max_attempts: int) -> Job:
        self._seq += 1
        job = Job(id=f"job-{self._seq}", document_id=document_id, stage=stage,
                  run_after=now, max_attempts=max_attempts)
        self.jobs[job.id] = job
        return job

    # --- producer ---
    def enqueue(self, document_id: str, now: float = 0.0,
                max_attempts: int = 5) -> Job:
        self.completed.setdefault(document_id, set())
        return self._new_job(document_id, "parse", now, max_attempts)

    # --- worker: claim one runnable job (SKIP LOCKED semantics) ---
    def claim(self, worker: str, now: float) -> Job | None:
        candidates = [j for j in self.jobs.values()
                      if j.state == "queued" and j.run_after <= now]
        if not candidates:
            return None
        job = sorted(candidates, key=lambda j: (j.run_after, j.id))[0]
        job.state = "running"
        job.locked_by = worker
        job.locked_at = now
        job.attempts += 1
        return job

    # --- worker: success ---
    def complete(self, job_id: str, now: float, output: object | None = None) -> None:
        job = self.jobs[job_id]
        if job.state != "running":
            raise RuntimeError(f"cannot complete job in state {job.state}")
        job.state = "done"
        self.completed[job.document_id].add(job.stage)
        if output is not None:
            # Idempotent upsert keyed by (document, stage): re-running a stage
            # REPLACES its output -> no duplicate chunks (TC-110).
            self.output[(job.document_id, job.stage)] = output
        nxt = _next_stage(job.stage)
        if nxt is not None:
            self._new_job(job.document_id, nxt, now, job.max_attempts)

    # --- worker: failure -> backoff or dead-letter ---
    def fail(self, job_id: str, now: float, error: str) -> None:
        job = self.jobs[job_id]
        if job.attempts >= job.max_attempts:
            job.state = "dead"
            job.last_error = error
            self.dead[job.document_id] = error          # ingest_status = failed
        else:
            delay = self.base_backoff * (2 ** (job.attempts - 1))
            job.state = "queued"
            job.run_after = now + delay
            job.locked_by = None
            job.locked_at = None
            job.last_error = error

    # --- reaper: reclaim leases from crashed workers (crash recovery) ---
    def reap(self, now: float) -> list[str]:
        reclaimed = []
        for job in self.jobs.values():
            if (job.state == "running" and job.locked_at is not None
                    and now - job.locked_at >= self.lease_timeout):
                job.state = "queued"
                job.locked_by = None
                job.locked_at = None
                reclaimed.append(job.id)
        return reclaimed

    # --- surfaced status (doc 11 §3.2) ---
    def document_status(self, document_id: str) -> str:
        if document_id in self.dead:
            return "failed"
        done = self.completed.get(document_id, set())
        if done.issuperset(STAGES):
            return "done"
        started = bool(done) or any(
            j.document_id == document_id and (j.attempts > 0 or j.stage != "parse")
            for j in self.jobs.values())
        return "processing" if started else "queued"

    def chunk_count(self, document_id: str) -> int:
        out = self.output.get((document_id, "embed"))
        return len(out) if isinstance(out, list) else 0
