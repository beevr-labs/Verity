#!/usr/bin/env python
"""BM3 — extractor quality benchmark: Qwen2.5-1.5B vs 3B (same chunks, same
NLI gate). Measures usefulness proxies on the real Cenveo credit agreement:
item count, % with party, % with date/trigger, mean length, fragment rate.

Usage: python scripts/bench_extractor.py [n_chunks]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

from beevr.agent import AgentRun
from beevr.audit import AuditLog
from beevr.ingest import ingest_document
from beevr.llm import LlmExtractor, TransformersLLM
from beevr.models import CrossEncoderNLI
from beevr.store import Session, Store

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]


def fragment(text: str) -> bool:
    w = text.split()
    return len(w) < 5 or (text and text[0].islower())


def main() -> int:
    nli = CrossEncoderNLI()
    text = (ROOT / "test-data" / "cenveo-credit-agreement.txt").read_text(encoding="utf-8")

    print(f"== BM3: extractor benchmark on first {N} chunks of the real "
          f"Cenveo agreement (same NLI gate) ==")
    for model_name in MODELS:
        store = Store(audit=AuditLog())
        store.put_matter("A", client="x", name="x")
        ingest_document(store, "A", "doc", text.encode(), "cenveo.txt")
        # keep only the first N chunks (identical subset for both models)
        keep = dict(list(store.chunks.items())[:N])
        store.chunks = keep

        t0 = time.time()
        llm = TransformersLLM(model_name)
        load_s = time.time() - t0

        run = AgentRun(run_id="bm", matter_id="A",
                       session=Session("bm", matter_grants=frozenset({"A"})),
                       store=store, nli=nli, extractor=LlmExtractor(llm))
        t1 = time.time()
        items = run.extract_phase()
        dt = time.time() - t1

        n = len(items)
        party = sum(1 for i in items if i.party.strip())
        due = sum(1 for i in items if i.trigger_or_due.strip())
        frag = sum(1 for i in items if fragment(i.text))
        avg_w = (sum(len(i.text.split()) for i in items) / n) if n else 0

        print(f"\n--- {model_name} (load {load_s:.0f}s, extract {dt:.0f}s, "
              f"{dt/N:.1f}s/chunk) ---")
        print(f"  items (NLI-verified): {n}")
        print(f"  with party          : {party}/{n}")
        print(f"  with date/trigger   : {due}/{n}")
        print(f"  mean length (words) : {avg_w:.1f}")
        print(f"  fragment rate       : {frag}/{n}")
        print("  samples:")
        for i in items[:5]:
            print(f"   [{i.item_type:<19}] {i.text[:88]}")

        # free VRAM before the next model
        del llm, run
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
