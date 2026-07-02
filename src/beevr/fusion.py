"""Hybrid retrieval fusion — doc 22 §2, FR-KM-15 (refines FR-KM-07).

Reciprocal Rank Fusion (RRF) over the vector and BM25 rankings — rank-based, so
no score normalization across engines is needed:

    score_rrf(chunk) = Σ_retriever  1 / (k_rrf + rank_retriever(chunk))

BM25 catches exact legal terms (defined terms, clause numbers, party names);
vectors catch paraphrase. RRF combines their rankings robustly and cheaply.
An optional reranker refines the short list afterwards (bge-reranker, doc 14).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fused:
    chunk_id: str
    score: float


def rrf_fuse(rankings: list[list[str]], *, k_rrf: int = 60, top_n: int = 8,
             allowed: set[str] | None = None) -> list[Fused]:
    """Fuse ranked lists of chunk_ids (each list already best-first).

    `allowed`: matter-partition guard — only chunk_ids in this set survive
    (isolation is enforced upstream too; this is defense in depth). If None,
    all chunks are allowed.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            if allowed is not None and chunk_id not in allowed:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Fused(cid, sc) for cid, sc in ordered[:top_n]]


def fuse_ids(rankings: list[list[str]], **kw) -> list[str]:
    return [f.chunk_id for f in rrf_fuse(rankings, **kw)]
