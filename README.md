# Verity

**Sovereign, auditable AI copilot for legal & compliance work.**

Verity lets an in-house legal & compliance team query and act on its documents
with data kept entirely inside the customer's boundary, every answer cited to a
verified source span, and every action audit-logged. Built for regulated
financial institutions where cloud AI is not an option.

## Trust guarantees (release-gated, tested)

Five promises encode the product. Each is a hard CI gate — a failure blocks
release regardless of other progress:

| Promise | Enforcement |
|---|---|
| **No fabricated citations** | Two-stage verification: locator grounding + NLI entailment, plus deterministic entity/role/template guards. Fabrication rate must be **0** on the adversarial eval set. |
| **Cross-matter isolation** | Single data-access chokepoint (deny-by-default RBAC + ethical walls) + Postgres row-level security as defense in depth. |
| **No ungoverned execution** | The LLM only *proposes*; a governance Kernel validates every action against a whitelist, budgets, and policy. Consequential actions pause at a human-in-the-loop checkpoint. |
| **Kill switch, no partial effects** | Per-run and global halt; idempotent, exactly-once execution. |
| **Zero egress** | Default-deny at the app layer (no outbound client) and the network layer (internal Docker network); a socket-capture proof harness asserts 0 outbound bytes on the full pipeline. |

Answers with insufficient evidence **abstain** — the system never guesses.

## Architecture (single-tenant, in-boundary)

```
browser SPA ── FastAPI (auth: OIDC RS256/JWKS · RBAC · doc-11 error model)
   │
   ├─ ingest: parsers (pdf/docx/eml/txt) · legal-aware chunker · OCR w/ bboxes
   ├─ retrieval: hybrid RRF (vector + BM25), matter-partitioned
   ├─ verification: NLI entailment + entity/role/template guards → cite or abstain
   ├─ agent: obligation extraction → Kernel → HITL → exactly-once persist
   ├─ audit: append-only, hash-chained, tamper-evident
   └─ models (local, pluggable): bge-m3 · DeBERTa NLI · Qwen instruct
```

Storage is Postgres-centric (pgvector + FTS + RLS) with MinIO for immutable
originals. Everything runs offline once model weights are local; the only
possible egress is an explicitly configured, redaction-gated escalation path
(off by default).

## Quickstart

```bash
pip install -e ".[dev]"          # core + tests (runs in stub-model mode)
python -m pytest                 # full suite incl. the 5 release gates
python scripts/egress_proof.py   # zero-egress proof harness

# run the API + SPA
python -m uvicorn beevr.api:create_app --factory --app-dir src --port 8777
# browse http://localhost:8777

# real model tier (GPU): bge-m3 + NLI cross-encoder + LLM extractor
pip install -e ".[models]"
```

### Docker (in-boundary stack)

```bash
docker compose up -d             # app + Postgres(pgvector) + MinIO + migrations
```

The internal network is `internal: true` — containers have no outbound route
(network-layer egress deny).

## Evaluation harness

Citation quality is measured, not assumed. The harness runs a golden set of
four classes (grounded / distractor / unanswerable / adversarial) over real
SEC-EDGAR legal documents and gates on fabrication:

```bash
python scripts/eval_harness.py                                # credit agreement
python scripts/eval_harness.py test-data/golden-guaranty.json # guaranty
python scripts/eval_harness.py test-data/golden-nda.json      # NDA
```

Current status: **35/35** across three real documents — fabrication 0.000,
citation precision 1.000, abstention correctness 1.000. (Dev-set scale; the
release bar is a ≥300-item lawyer-labeled set run through the same gate.)

## Repository layout

```
src/beevr/        core modules (locator, audit, kernel, verification, fusion,
                  queue, redaction, store, egress, pipeline, ingest, models,
                  llm, agent, sso, api) + static SPA
tests/            pytest suite; release gates marked `release_gating`
scripts/          egress_proof.py · eval_harness.py · run_real_doc.py
db/migrations/    Postgres schema + row-level security (forward-only)
test-data/        real EDGAR corpora + golden evaluation sets
```

## License

Proprietary — © BeevR. All rights reserved.
