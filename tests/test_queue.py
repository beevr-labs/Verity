"""Ingest job queue tests — FR-KM-14, TC-110 (crash recovery, no dup chunks),
TC-111 (dead-letter + failed surfacing, batch survives)."""
from beevr.queue import STAGES, IngestQueue


def _run_stage(q: IngestQueue, worker: str, now: float, output=None):
    job = q.claim(worker, now)
    assert job is not None
    q.complete(job.id, now, output=output)
    return job


def test_happy_path_reaches_done():
    q = IngestQueue()
    q.enqueue("doc-1", now=0)
    for stage in STAGES:
        out = ["ch1", "ch2", "ch3"] if stage == "embed" else None
        job = _run_stage(q, "w1", now=1, output=out)
        assert job.stage == stage
    assert q.document_status("doc-1") == "done"
    assert q.chunk_count("doc-1") == 3


def test_crash_recovery_no_duplicate_chunks_TC110():
    q = IngestQueue(lease_timeout=900)
    q.enqueue("doc-1", now=0)
    # parse, ocr, chunk succeed
    for _ in range(3):
        _run_stage(q, "w1", now=1)
    # embed: worker w1 claims then CRASHES (never completes)
    embed_job = q.claim("w1", now=10)
    assert embed_job.stage == "embed" and embed_job.state == "running"
    # reaper reclaims the stale lease after the timeout
    reclaimed = q.reap(now=10 + 900)
    assert embed_job.id in reclaimed and embed_job.state == "queued"
    # worker w2 re-runs embed and completes once
    job = q.claim("w2", now=1000)
    assert job.id == embed_job.id and job.attempts == 2   # re-attempt
    q.complete(job.id, now=1000, output=["ch1", "ch2", "ch3"])
    # extract_graph
    _run_stage(q, "w2", now=1001)
    assert q.document_status("doc-1") == "done"
    assert q.chunk_count("doc-1") == 3                     # NOT 6 — no duplicates


def test_idempotent_stage_output_upsert():
    q = IngestQueue()
    q.enqueue("doc-1", now=0)
    for _ in range(3):
        _run_stage(q, "w1", now=1)
    job = q.claim("w1", now=1)
    q.complete(job.id, now=1, output=["a", "b"])
    # simulate a redundant re-complete of the same stage output -> replace
    q.output[("doc-1", "embed")] = ["a", "b"]
    assert q.chunk_count("doc-1") == 2


def test_dead_letter_after_max_attempts_and_batch_survives_TC111():
    q = IngestQueue(base_backoff=5)
    q.enqueue("bad", now=0, max_attempts=3)
    # 'good' becomes runnable only later, so 'bad' is processed to death first
    # (the queue fairly prefers the earliest run_after, which is realistic).
    q.enqueue("good", now=1_000_000, max_attempts=3)

    # 'bad' fails its parse stage until it dead-letters
    now = 0.0
    for _ in range(3):
        job = q.claim("w1", now)
        assert job.document_id == "bad"
        q.fail(job.id, now, error="corrupt PDF header")
        now += 100  # past the (≤20s) backoff, still before 'good' is runnable

    assert q.document_status("bad") == "failed"
    dead = [j for j in q.jobs.values() if j.document_id == "bad"][0]
    assert dead.state == "dead" and "corrupt PDF header" in dead.last_error

    # the rest of the batch still completes
    now = 1_000_000
    for stage in STAGES:
        job = q.claim("w2", now)
        assert job.document_id == "good"
        q.complete(job.id, now)
    assert q.document_status("good") == "done"


def test_backoff_delays_requeue():
    q = IngestQueue(base_backoff=5)
    q.enqueue("doc-1", now=0, max_attempts=5)
    job = q.claim("w1", now=0)
    q.fail(job.id, now=0, error="transient")
    # after 1st failure (attempts=1) delay = 5 * 2^0 = 5 -> not runnable at t=1
    assert q.claim("w1", now=1) is None
    assert q.claim("w1", now=5) is not None
