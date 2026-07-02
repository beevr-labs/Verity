#!/usr/bin/env python
"""Run a REAL legal document (SEC EDGAR credit agreement) through the full
pipeline with REAL models: ingest -> matter Q&A (NLI-verified citations,
abstain-over-guess) -> obligation extraction (Qwen LLM -> NLI gate -> HITL).

Honest report: prints what verified, what abstained, what was extracted.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from beevr.api import AppState, create_app


def main() -> int:
    t0 = time.time()
    print("== loading real models (bge-m3 + DeBERTa NLI + Qwen2.5-1.5B on CUDA) ==")
    state = AppState.with_real_models()
    print(f"   model_mode={state.model_mode}  ({time.time()-t0:.0f}s)")

    client = TestClient(create_app(state))
    tok = client.post("/auth/token", json={
        "sub": "counsel@fi.example", "roles": ["user"],
        "matter_grants": ["CENVEO"]}).json()["token"]
    hdr = {"Authorization": f"Bearer {tok}"}
    client.post("/matters", json={"id": "CENVEO", "client": "Bank of America",
                                  "name": "Cenveo Incremental Term Loan"}, headers=hdr)

    text = (ROOT / "test-data" / "cenveo-credit-agreement.txt").read_text(encoding="utf-8")
    r = client.post("/matters/CENVEO/documents", json={
        "filename": "cenveo-credit-agreement.txt", "content": text}, headers=hdr).json()
    print(f"\n== ingested real document: {r['chunks']} chunks ==")

    # ---- Matter Q&A: real questions a lawyer would ask -----------------------
    QUESTIONS = [
        # answerable — assertion-style, grounded in the document
        "Bank of America, N.A. is the Administrative Agent under the Credit Agreement",
        "the Borrower is Cenveo Corporation, a Delaware corporation",
        "the Credit Agreement Supplement is dated as of June 5, 2012",
        # unanswerable — NOT in this document; must abstain
        "the interest rate is fixed at nine percent per annum",
        # adversarial trap — invites a fabricated citation
        "Section 99.9 requires the Borrower to deliver quarterly ESG reports",
    ]
    print("\n== matter Q&A (every shown citation is NLI-verified; else abstain) ==")
    for q in QUESTIONS:
        a = client.post("/matters/CENVEO/queries", json={"question": q}, headers=hdr).json()
        if a["abstained"]:
            print(f"  ABSTAIN  (insufficient evidence) <- {q[:70]}")
        else:
            cit = a["claims"][0]["citations"][0]
            print(f"  ANSWER   conf={a['confidence']:<6} cite={cit['document_id']}"
                  f" verified={cit['verified']} <- {q[:70]}")
            print(f"           span: \"{cit['snippet'][:90]}...\"")

    # ---- Obligation extraction (Qwen -> NLI gate -> HITL) --------------------
    print("\n== obligation & covenant extraction (real LLM, NLI-gated) ==")
    t1 = time.time()
    run = client.post("/matters/CENVEO/agent/runs",
                      json={"workflow": "obligation_extraction"}, headers=hdr).json()
    props = run["proposals"]
    print(f"   {len(props)} proposals in {time.time()-t1:.0f}s (all NLI-verified)")
    for p in props[:12]:
        print(f"   [{p['item_type']:<20}] {p['text'][:95]}")
        print(f"     party={p['party'] or '-':<22} due={p['trigger_or_due'] or '-'}"
              f"  cite={p['citation']['document_id']} v={p['verified']}")
    if len(props) > 12:
        print(f"   ... and {len(props)-12} more")

    # HITL: approve the first item, prove exactly-once + audit
    if props:
        client.post(f"/agent/proposals/{run['run_id']}/0/decision",
                    json={"decision": "approve"},
                    headers={**hdr, "Idempotency-Key": "real-0"})
        d = client.get(f"/agent/runs/{run['run_id']}", headers=hdr).json()
        print(f"\n   HITL: approved item 0 -> persisted={len(d['persisted'])}"
              f" (cited verified={d['persisted'][0]['citation']['verified']})")

    ev = client.get("/matters/CENVEO/audit", headers=hdr).json()["events"]
    print(f"   audit events: {len(ev)} (hash-chained)")
    print(f"\n== done in {time.time()-t0:.0f}s ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
