"""Generative matter Q&A — the "answer service" of doc 02 §4.1, completed.

The lawyer asks an OPEN question ("What are the conditions to the Incremental
Term Loans?"). Flow:

  1. retrieve      — RRF over the matter partition (isolation enforced)
  2. compose (LLM) — the model answers ONLY from numbered excerpts and returns
                     JSON claims, each naming its source excerpt(s)
                     (doc 15 §6 discipline: no outside knowledge, no advice)
  3. verify        — every claim runs the FULL citation gate (clause premises,
                     entity/role/template guards, NLI, τ pinned) against the
                     excerpts it cited; unverified claims are DROPPED
  4. assemble      — surviving claims become the cited answer; none -> ABSTAIN

The LLM can only make the answer BETTER-said, never differently-true: whatever
it writes must still be entailed by a real span or it never reaches the user.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .fusion import fuse_ids
from .locator import Locator
from .store import Session, Store
from .verification import NLI, ShownCitation, Verifier

COMPOSE_PROMPT = """You are a legal research assistant inside a compliance tool. \
Answer the QUESTION using ONLY the numbered EXCERPTS from the matter's documents.

Rules (strict):
- Output ONLY JSON: {{"answer_possible": true/false,
  "claims": [{{"text": "<one factual statement>", "sources": [<excerpt numbers>]}}]}}
- Every claim MUST be directly supported by its cited excerpt(s), faithful to
  their wording. NEVER use outside knowledge, NEVER guess, NEVER give legal
  advice or conclusions — restate what the documents say.
- Definitions, recitals, preambles and schedules COUNT as answers: if the
  excerpt states or defines the fact asked about (e.g. "X means ... dated
  <date>" answers a when-question; "Y, as Administrative Agent" answers a
  who-question), extract it.
- If the excerpts do not answer the question: {{"answer_possible": false, "claims": []}}

QUESTION: {question}

EXCERPTS:
{excerpts}

JSON:"""

_JSON_OBJ = re.compile(r"\{.*\}", re.S)
# Schema-aware fallback for single-character JSON slips from small LLMs
# (measured: Qwen-1.5B emitted `"sources":[1]]}}` — right answer, broken JSON).
# Verification still gates every claim; this only recovers the *parse*.
_CLAIM_RX = re.compile(
    r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"sources"\s*:\s*\[([^\]]*)\]', re.S)
_POSSIBLE_RX = re.compile(r'"answer_possible"\s*:\s*true', re.I)


def _parse_compose(raw: str) -> dict | None:
    m = _JSON_OBJ.search(raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    if not _POSSIBLE_RX.search(raw):
        return None
    claims = []
    for cm in _CLAIM_RX.finditer(raw):
        text = cm.group(1).replace('\\"', '"').strip()
        srcs = [int(n) for n in re.findall(r"\d+", cm.group(2))]
        if text and srcs:
            claims.append({"text": text, "sources": srcs})
    return {"answer_possible": True, "claims": claims} if claims else None


@dataclass
class ComposedClaim:
    text: str
    citations: list[ShownCitation] = field(default_factory=list)


@dataclass
class ComposedAnswer:
    abstained: bool
    confidence: str
    answer_text: str
    claims: list[ComposedClaim] = field(default_factory=list)


# stopwords pollute the lexical arm at scale (1,900+ chunks all matching
# "the/is/under" outranked the defining recital on the 357-page stress case)
_STOP = frozenset("a an and are as at be by for from has have in is it of on or "
                  "that the this to under was were what when which who whom why "
                  "how does do did".split())


def matter_ask(store: Store, session: Session, matter_id: str, question: str,
               *, llm, nli: NLI, embedder=None, top_n: int = 6,
               ts: str = "") -> ComposedAnswer:
    # 1. retrieve (isolation + RRF; real embeddings when available)
    rows = store.retrieve_chunks(session, matter_id, ts=ts)
    q_tokens = {t.lower().strip("?,.") for t in question.split()} - _STOP

    def overlap(r):
        return len(q_tokens & {t.lower() for t in r.data["text"].split()})

    if embedder is not None and rows and rows[0].data.get("embedding"):
        qv = embedder.embed(question)
        def cos(r):
            v = r.data["embedding"]
            num = sum(a * b for a, b in zip(qv, v))
            den = (sum(a * a for a in qv) ** 0.5) * (sum(b * b for b in v) ** 0.5)
            return num / den if den else 0.0
        vec = [r.id for r in sorted(rows, key=lambda r: -cos(r))]
    else:
        vec = [r.id for r in sorted(rows, key=lambda r: -overlap(r))]
    lex = [r.id for r in sorted(rows, key=lambda r: -overlap(r)) if overlap(r) > 0]
    allowed = store.resolve_candidates(session, matter_id, [r.id for r in rows], ts=ts)
    top = fuse_ids([vec, lex], allowed=allowed, top_n=top_n)

    by_id = {r.id: r.data["text"] for r in rows}
    excerpts = [(i + 1, cid, by_id[cid]) for i, cid in enumerate(top)]
    if not excerpts:
        return ComposedAnswer(True, "insufficient", "")

    # 2+3. MAP-THEN-VERIFY: the LLM reads ONE excerpt at a time (measured on the
    # 357-page stress case: with 6 dense excerpts in one prompt both 1.5B and 3B
    # refuse even when the answer is verbatim in excerpt [2]; with a single
    # excerpt they answer instantly). Per-excerpt claims then run the full
    # verification gate; duplicates dedupe on normalized text.
    verifier = Verifier(by_id, nli, tau_e=0.92, clause_premises=True)  # see pipeline.py
    survivors: list[ComposedClaim] = []
    seen: set[str] = set()
    proposed = 0
    prompts = [COMPOSE_PROMPT.format(question=question, excerpts=f"[1] {ex_text}")
               for _, _, ex_text in excerpts]
    if hasattr(llm, "generate_batch"):
        raws = llm.generate_batch(prompts, max_new_tokens=400)
    else:
        raws = [llm.generate(p, max_new_tokens=400) for p in prompts]
    for (_, cid, ex_text), raw in zip(excerpts, raws):
        data = _parse_compose(raw)
        if not data or not data.get("answer_possible") \
                or not isinstance(data.get("claims"), list):
            continue
        for cl in data["claims"]:
            text = str(cl.get("text", "")).strip()
            srcs = [int(n) for n in (cl.get("sources") or [])
                    if isinstance(n, int) or str(n).isdigit()]
            if not text or 1 not in srcs:
                continue                                 # must cite THIS excerpt
            proposed += 1
            key = re.sub(r"\W+", "", text.lower())
            if key in seen:
                continue
            loc = Locator(cid, "pdf", char_range=(0, len(by_id[cid])), page=1)
            if verifier.verify_pair(text, loc).passed:
                seen.add(key)
                survivors.append(ComposedClaim(
                    text, [ShownCitation(loc, by_id[cid])]))

    # 4. assemble-or-abstain
    if not survivors:
        return ComposedAnswer(True, "insufficient", "")
    if store.audit is not None:
        store.audit.append(actor=session.user_id, actor_kind="user", type="query",
                           target=matter_id, payload=question, ts=ts)
    conf = "high" if len(survivors) == proposed else "medium"
    return ComposedAnswer(False, conf, " ".join(c.text for c in survivors), survivors)
