"""Reference matter-Q&A pipeline — ties the offline cores together (doc 02 §4.1).

retrieve (scoped) -> hybrid fuse -> verify citations -> assemble-or-abstain -> audit.

Deliberately in-boundary: it touches only the local Store, local fusion, and the
injected NLI. It opens no network sockets — which is what the egress proof
(TC-501) asserts. Production swaps the simulated retrieval arms for real
pgvector + BM25 and the stub NLI for a local cross-encoder, without changing the
control flow.
"""
from __future__ import annotations

from .fusion import fuse_ids
from .locator import Locator
from .store import Session, Store
from .verification import NLI, AnswerResult, Claim, Verifier


def matter_qa(store: Store, session: Session, matter_id: str, question: str,
              *, nli: NLI, embedder=None, ts: str = "") -> AnswerResult:
    # 1. retrieve candidates within the authorized matter partition (isolation)
    rows = store.retrieve_chunks(session, matter_id, ts=ts)
    q_tokens = {t.lower() for t in question.split()}

    def overlap(text: str) -> int:
        return len(q_tokens & {t.lower() for t in text.split()})

    # vector arm: REAL cosine over ingested embeddings when an embedder is
    # available (bge-m3 in production, doc 14); word-overlap fallback otherwise
    if embedder is not None and rows and rows[0].data.get("embedding"):
        qv = embedder.embed(question)
        def cos(r):
            v = r.data["embedding"]
            num = sum(a * b for a, b in zip(qv, v))
            den = (sum(a * a for a in qv) ** 0.5) * (sum(b * b for b in v) ** 0.5)
            return num / den if den else 0.0
        vector_arm = [r.id for r in sorted(rows, key=lambda r: -cos(r))]
    else:
        vector_arm = [r.id for r in sorted(rows, key=lambda r: -overlap(r.data["text"]))]
    bm25_arm = [r.id for r in sorted(rows, key=lambda r: -overlap(r.data["text"]))
                if overlap(r.data["text"]) > 0]               # lexical hits

    # 2. matter-partition guard feeds fusion (defense in depth)
    allowed = store.resolve_candidates(session, matter_id, [r.id for r in rows], ts=ts)
    top_ids = fuse_ids([vector_arm, bm25_arm], allowed=allowed, top_n=5)

    # 3. build the claim + candidate citations (each chunk as its own source span)
    documents = {r.id: r.data["text"] for r in rows}
    citations = [Locator(cid, "pdf", char_range=(0, len(documents[cid])), page=1)
                 for cid in top_ids]
    claim = Claim(question, citations=citations)

    # 4. verify (Stage A grounding + Stage B entailment) and assemble-or-abstain
    # Tuned per doc 13 §3.4 on the EDGAR dev set (see PROGRESS): clause-granularity
    # premises (party-boundary segmentation) + tau_e=0.97 separate the
    # entity-confusion trap (best-clause 0.9487) from grounded claims (>=0.9968).
    # Pinned per release; MUST be re-tuned when the golden set scales (doc 13 §3.1).
    result = Verifier(documents, nli, tau_e=0.97,
                      clause_premises=True).assemble([claim], needs_evidence=True)

    # 5. audit the answer-producing action
    if store.audit is not None:
        store.audit.append(actor=session.user_id, actor_kind="user", type="query",
                           target=matter_id, payload=question, ts=ts)
    return result
