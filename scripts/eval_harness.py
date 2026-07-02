#!/usr/bin/env python
"""Citation-eval harness — doc 13 §3 (closes the OI-1 spec in code).

Runs a golden set (4 classes: answerable-grounded, answerable-distractor,
unanswerable, adversarial) through the REAL pipeline (real NLI verifier) and
reports the doc-13 §3.3 metrics:

  * fabricated-citation rate  — RELEASE-GATING: must be 0 on the adversarial
    class (any citation shown for an abstain-class item, or a grounded item
    cited to a span missing its gold keywords, counts as fabricated)
  * citation precision        — verified citations containing gold keywords
  * abstention correctness    — abstain-class items that actually abstained
  * over-abstention           — grounded items wrongly abstained (guardrail)

Exit code 1 if the adversarial gate fails -> pluggable as the CI gate
(doc 19 §5 "eval harness" stage; doc 13 §3.4).

Usage:  python scripts/eval_harness.py [golden.json]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    golden_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "test-data" / "golden-cenveo.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    print("== eval harness (doc 13 §3) ==")
    print(f"   golden set: {golden_path.name} · {len(golden['items'])} items")

    t0 = time.time()
    try:
        from beevr.models import CrossEncoderNLI
        nli = CrossEncoderNLI()
        print(f"   verifier: REAL NLI cross-encoder ({time.time()-t0:.0f}s to load)")
    except Exception as ex:
        print(f"   verifier: NLI unavailable ({ex}); using lexical stub — "
              f"results are NOT release-grade")
        from beevr.verification import lexical_overlap
        nli = lambda span, claim: lexical_overlap(claim, span)

    from beevr.audit import AuditLog
    from beevr.ingest import ingest_document
    from beevr.pipeline import matter_qa
    from beevr.store import Session, Store

    store = Store(audit=AuditLog())
    store.put_matter(golden["matter"], client="eval", name="eval")
    text = (ROOT / "test-data" / golden["corpus"]).read_text(encoding="utf-8")
    rep = ingest_document(store, golden["matter"], "doc-eval",
                          text.encode(), golden["corpus"])
    session = Session("eval", matter_grants=frozenset({golden["matter"]}))
    print(f"   corpus: {golden['corpus']} -> {rep.chunks} chunks\n")

    per_class: dict[str, list] = {}
    fabricated = 0
    shown_citations = 0
    good_citations = 0
    grounded_abstained = 0
    grounded_total = 0
    abstain_ok = 0
    abstain_total = 0

    for item in golden["items"]:
        res = matter_qa(store, session, golden["matter"], item["claim"], nli=nli)
        cited_spans = [c.snippet for claim in res.claims for c in claim.citations]
        shown_citations += len(cited_spans)
        expect = item["expect"]
        cls = item["class"]

        if expect == "abstain":
            abstain_total += 1
            ok = res.abstained
            if ok:
                abstain_ok += 1
            else:
                fabricated += len(cited_spans)   # any citation here is fabricated
        else:
            grounded_total += 1
            if res.abstained:
                grounded_abstained += 1
                ok = False
            else:
                gold = item.get("gold_keywords", [])
                supported = any(all(k.lower() in s.lower() for k in gold)
                                for s in cited_spans)
                good_citations += sum(
                    1 for s in cited_spans
                    if all(k.lower() in s.lower() for k in gold))
                if not supported:
                    fabricated += len(cited_spans)  # cited, but not to the gold span
                ok = supported
        mark = "PASS" if ok else "FAIL"
        per_class.setdefault(cls, []).append(mark)
        print(f"  [{mark}] {item['id']:<3} {cls:<22} "
              f"{'abstained' if res.abstained else f'{len(cited_spans)} cite(s)':<12} "
              f"{item['claim'][:58]}")

    # ---- metrics (doc 13 §3.3) ------------------------------------------------
    fab_rate = fabricated / shown_citations if shown_citations else 0.0
    precision = good_citations / shown_citations if shown_citations else 1.0
    abst_correct = abstain_ok / abstain_total if abstain_total else 1.0
    over_abst = grounded_abstained / grounded_total if grounded_total else 0.0

    adversarial_fails = per_class.get("adversarial", []).count("FAIL")

    print("\n== metrics ==")
    print(f"   fabricated-citation rate : {fab_rate:.3f}  (gate: 0 on adversarial)")
    print(f"   citation precision       : {precision:.3f}  (target >= 0.95, NFR-02)")
    print(f"   abstention correctness   : {abst_correct:.3f}  (target >= 0.95)")
    print(f"   over-abstention          : {over_abst:.3f}  (guardrail — monitored)")
    for cls, marks in per_class.items():
        print(f"   {cls:<24}: {marks.count('PASS')}/{len(marks)} pass")

    if adversarial_fails:
        print("\nGATE: FAIL — adversarial fabrication detected; release blocked (doc 13 §3.4)")
        return 1
    print("\nGATE: PASS — 0 fabricated citations on the adversarial class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
