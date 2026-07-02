"""RRF fusion tests — FR-KM-15, TC-112.

Exact-term queries surface via the BM25 arm; paraphrase via the vector arm;
RRF puts the gold chunk in the final top-n either way.
"""
from beevr.fusion import fuse_ids, rrf_fuse


def test_exact_term_query_surfaces_via_bm25_arm_TC112():
    gold = "chunk-gold"
    # Exact-term query (e.g. a defined term / clause number): BM25 ranks gold #1;
    # the vector arm suffers semantic drift and misses it entirely.
    vector = ["c1", "c2", "c3", "c4"]
    bm25 = [gold, "c5", "c6"]
    top = fuse_ids([vector, bm25], top_n=3)
    assert gold in top


def test_paraphrase_query_surfaces_via_vector_arm_TC112():
    gold = "chunk-gold"
    # Paraphrase query: the vector arm ranks gold #1; BM25 (no lexical overlap)
    # misses it. RRF still surfaces gold.
    vector = [gold, "c1", "c2"]
    bm25 = ["c4", "c5", "c6", "c7"]
    top = fuse_ids([vector, bm25], top_n=3)
    assert gold in top


def test_agreement_across_arms_beats_single_arm():
    # A chunk ranked well by BOTH arms should outrank one ranked well by only one.
    both = "chunk-both"
    one = "chunk-one"
    vector = [both, "x1", "x2", one]
    bm25 = [both, "y1", "y2", "y3"]
    fused = rrf_fuse([vector, bm25], top_n=10)   # keep all chunks to compare ranks
    order = [f.chunk_id for f in fused]
    assert order[0] == both
    assert order.index(both) < order.index(one)


def test_matter_partition_guard_filters_foreign_chunks():
    vector = ["a", "foreign", "b"]
    bm25 = ["foreign", "a", "b"]
    top = fuse_ids([vector, bm25], allowed={"a", "b"}, top_n=5)
    assert "foreign" not in top
    assert set(top) == {"a", "b"}
