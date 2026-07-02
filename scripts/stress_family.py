#!/usr/bin/env python
"""Amendment-chain stress test — a REAL document family in ONE matter:

  * Second A&R Credit Agreement (2014)  — the base,   ~672 pages
  * Amendment No. 1 (2016)              —            ~1,302 pages
  * Amendment No. 3 (2017)              —              ~357 pages
  (TransDigm / Credit Suisse, SEC EDGAR — ~2,300 pages total)

This is how contracts actually live: the current state of any clause is the
BASE as modified by the CHAIN. Questions are cross-document — including the
assignment question that correctly abstained when the matter held only
Amendment 3 (the answer lives in the base agreement).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datetime import date

from beevr.audit import AuditLog
from beevr.briefing import build_briefing
from beevr.compose import matter_ask
from beevr.ingest import ingest_document
from beevr.store import Session, Store

FAMILY = [
    ("base-2014", "transdigm-base-2014.txt"),
    ("amend1-2016", "transdigm-amend1-2016.txt"),
    ("amend3-2017", "transdigm-credit-agreement.txt"),
]

QUESTIONS = [
    # the question that FAILED on the single-amendment matter — answer in base §9
    ("Can a lender assign its loans, and is any consent required?", ["assign"]),
    ("What events constitute an event of default?", ["event of default"]),
    ("Who is the administrative agent?", ["administrative agent"]),
    # amendment-specific fact (answer only in Amendment 3)
    ("What is the aggregate principal amount of the Tranche G term loans?",
     ["1,819,000,000"]),
    # base-specific definition
    ("What does the term Material Adverse Effect mean?", ["material adverse effect"]),
    # cross-document: base date + latest amendment date
    ("When was the original credit agreement dated?", ["june 4, 2014"]),
    # unanswerable — expect abstain
    ("What is the LIBOR successor rate fallback for 2028?", None),
]


def main() -> int:
    t_all = time.time()
    print("== AMENDMENT-CHAIN STRESS: 3 real documents, one matter ==")

    from beevr.llm import TransformersLLM
    from beevr.models import Bgem3Embedder, BgeReranker, CrossEncoderNLI
    t0 = time.time()
    embedder = Bgem3Embedder()
    nli = CrossEncoderNLI()
    llm = TransformersLLM("Qwen/Qwen2.5-3B-Instruct")
    try:
        reranker = BgeReranker()
        print(f"   models loaded in {time.time()-t0:.0f}s (with reranker)")
    except Exception as ex:
        reranker = None
        print(f"   models loaded in {time.time()-t0:.0f}s (reranker unavailable: {ex})")

    store = Store(audit=AuditLog())
    store.put_matter("TDFAM", client="Credit Suisse AG", name="TransDigm facility")
    s = Session("counsel", matter_grants=frozenset({"TDFAM"}))

    total_chars = 0
    t0 = time.time()
    for doc_id, fn in FAMILY:
        text = (ROOT / "test-data" / fn).read_text(encoding="utf-8")
        total_chars += len(text)
        rep = ingest_document(store, "TDFAM", doc_id, text.encode(), fn,
                              embedder=embedder)
        print(f"   {doc_id:<12} {len(text):>9,} chars -> {rep.chunks:,} chunks")
    n_chunks = len(store.chunks)
    print(f"1. INGEST: {total_chars:,} chars (~{total_chars//1800:,} pages) -> "
          f"{n_chunks:,} chunks in {time.time()-t0:.0f}s")

    t0 = time.time()
    b = build_briefing(store, s, "TDFAM", today=date(2026, 7, 3))
    print(f"\n2. BRIEFING in {time.time()-t0:.1f}s: {len(b.deadlines)} dated items, "
          f"{len(b.suggested_questions)} suggested questions")

    print("\n3. CROSS-DOCUMENT Q&A:")
    low = {doc_id: (ROOT / "test-data" / fn).read_text(encoding="utf-8").lower()
           for doc_id, fn in FAMILY}
    lat = []
    for q, expect_kw in QUESTIONS:
        t0 = time.time()
        ans = matter_ask(store, s, "TDFAM", q, llm=llm, nli=nli,
                         embedder=embedder, reranker=reranker)
        dt = time.time() - t0
        lat.append(dt)
        if ans.abstained:
            verdict = "OK (expected abstain)" if expect_kw is None else "OVER-ABSTAIN"
            print(f"   [{dt:5.1f}s] ABSTAIN  {verdict:<22} <- {q[:58]}")
        else:
            src_docs = sorted({c.locator.document_id.split('-c')[0]
                               for cl in ans.claims for c in cl.citations})
            grep_ok = expect_kw is None or all(
                any(k.lower() in low[d] for d in low) for k in expect_kw)
            verdict = "grep-confirmed" if grep_ok else "check"
            print(f"   [{dt:5.1f}s] ANSWER   {verdict:<15} src={','.join(src_docs)} <- {q[:55]}")
            print(f"            \"{ans.answer_text[:105]}\"")
    lat.sort()
    print(f"   latency p50={lat[len(lat)//2]:.1f}s max={lat[-1]:.1f}s")
    print(f"\n== done in {time.time()-t_all:.0f}s ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
