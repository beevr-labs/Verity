"""Model layer tests — doc 14.

Router (FR-ML-02, TC-903): pure logic, always runs.
Real-model tests (bge-m3, DeBERTa NLI): skip unless torch+sentence-transformers
are installed and weights are present; CI-with-GPU runs them for real.
"""
import pytest

from beevr.models import ModelRouter

# ---- router: TC-903 --------------------------------------------------------
def test_short_factual_routes_fast_TC903():
    r = ModelRouter().route(task="qa", question="What is the termination date?")
    assert r.model == "fast"


def test_obligation_extraction_routes_strong_TC903():
    r = ModelRouter().route(task="obligation_extraction")
    assert r.model == "strong" and "obligation_extraction" in r.reason


def test_multi_doc_and_retry_route_strong_TC903():
    assert ModelRouter().route(task="qa", doc_count=5).model == "strong"
    assert ModelRouter().route(task="qa", retry=True).reason == "low_confidence_retry"


def test_long_question_routes_strong():
    q = " ".join(["word"] * 50)
    assert ModelRouter().route(task="qa", question=q).model == "strong"


# ---- real models (skip when runtime absent) --------------------------------
real = pytest.mark.skipif(
    not pytest.importorskip("importlib.util").find_spec("sentence_transformers"),
    reason="sentence-transformers not installed")


@pytest.fixture(scope="module")
def nli():
    st = pytest.importorskip("sentence_transformers")
    from beevr.models import CrossEncoderNLI
    try:
        return CrossEncoderNLI()
    except Exception as ex:                      # no weights / no network
        pytest.skip(f"NLI weights unavailable: {ex}")


@pytest.fixture(scope="module")
def embedder():
    st = pytest.importorskip("sentence_transformers")
    from beevr.models import Bgem3Embedder
    try:
        return Bgem3Embedder()
    except Exception as ex:
        pytest.skip(f"bge-m3 weights unavailable: {ex}")


def test_real_nli_entails_supported_claim(nli):
    span = "The Borrower shall maintain a leverage ratio below 3.0x at all times."
    assert nli(span, "The Borrower must keep its leverage ratio under 3.0x.") > 0.85
    # and rejects an unsupported one
    assert nli(span, "The Borrower may terminate the agreement at will.") < 0.5


def test_real_nli_deterministic(nli):
    span = "This Agreement terminates on 31 March 2027."
    claim = "The agreement ends on 31 March 2027."
    assert nli(span, claim) == nli(span, claim)


def test_real_bge_m3_semantic_neighbors(embedder):
    assert embedder.dim == 1024                          # doc 14 / schema VECTOR(1024)
    import numpy as np
    v_q = np.array(embedder.embed("What leverage ratio must the borrower keep?"))
    v_gold = np.array(embedder.embed(
        "The Borrower shall maintain a leverage ratio below 3.0x."))
    v_noise = np.array(embedder.embed("The office cafeteria closes at 3pm."))
    assert v_q @ v_gold > v_q @ v_noise                  # paraphrase beats noise


def test_real_verification_pipeline_with_real_nli(nli):
    """TC-205-class: the two-stage verifier with the REAL NLI drops the
    unsupported claim and keeps the supported one."""
    from beevr.locator import Locator
    from beevr.verification import Claim, Verifier
    text = "The Borrower shall maintain a leverage ratio below 3.0x at all times."
    docs = {"agr": text}
    v = Verifier(docs, nli, tau_e=0.85, tau_l=0.2)
    span = (0, len(text))                     # exact span; (0, len+2) would fail Stage A
    good = Claim("The Borrower must maintain a leverage ratio below 3.0x",
                 citations=[Locator("agr", "pdf", char_range=span, page=1)])
    bad = Claim("The Borrower may sell collateral without notice",
                citations=[Locator("agr", "pdf", char_range=span, page=1)])
    res = v.assemble([good, bad])
    kept = [c.text for c in res.claims]
    assert any("leverage ratio" in t for t in kept)
    assert not any("collateral" in t for t in kept)
