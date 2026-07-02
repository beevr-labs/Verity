"""Generative Q&A tests — compose.py (the doc 02 §4.1 answer service).

Offline: a FakeLLM proves the verify gate governs whatever the model writes —
supported claims survive, unsupported/fabricated ones are dropped, refusals and
garbage abstain. Real: Qwen answers an open question on the Cenveo agreement.
"""
import json

import pytest

from beevr.audit import AuditLog
from beevr.compose import matter_ask
from beevr.store import Session, Store
from beevr.verification import lexical_overlap


def _nli(span: str, claim: str) -> float:
    return lexical_overlap(claim, span)


def _store():
    s = Store(audit=AuditLog())
    s.put_matter("A", client="X", name="Facility A")
    s.put_chunk("c0", "A", text="The Borrower shall maintain a leverage ratio below 3.0x at all times.")
    s.put_chunk("c1", "A", text="This Agreement terminates on 31 March 2027 unless renewed in writing.")
    return s, Session("alice", matter_grants=frozenset({"A"}))


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)

    def generate(self, prompt, *, max_new_tokens=600):
        assert "EXCERPTS" in prompt and "ONLY" in prompt   # discipline present
        return self.reply


def test_supported_claim_survives_with_citation():
    store, s = _store()
    ans = matter_ask(store, s, "A", "leverage ratio the Borrower shall maintain",
                     llm=FakeLLM({"answer_possible": True, "claims": [
                         {"text": "The Borrower shall maintain a leverage ratio below 3.0x",
                          "sources": [1]}]}),
                     nli=_nli)
    assert not ans.abstained
    assert "leverage ratio" in ans.answer_text
    assert ans.claims[0].citations[0].verified


def test_fabricated_claim_dropped_then_abstain():
    store, s = _store()
    # LLM invents a fact and cites excerpt 1 — verification must kill it
    ans = matter_ask(store, s, "A", "leverage ratio the Borrower shall maintain",
                     llm=FakeLLM({"answer_possible": True, "claims": [
                         {"text": "The Borrower may pledge its aircraft fleet as collateral",
                          "sources": [1]}]}),
                     nli=_nli)
    assert ans.abstained and ans.confidence == "insufficient"


def test_mixed_claims_partial_survival():
    store, s = _store()
    ans = matter_ask(store, s, "A", "leverage ratio the Borrower shall maintain and termination",
                     llm=FakeLLM({"answer_possible": True, "claims": [
                         {"text": "The Borrower shall maintain a leverage ratio below 3.0x",
                          "sources": [1, 2]},
                         {"text": "The facility is secured by real estate in Texas",
                          "sources": [2]}]}),
                     nli=_nli)
    assert not ans.abstained
    assert len(ans.claims) == 1                        # fabricated one dropped
    assert ans.confidence == "medium"                  # partial survival


def test_llm_refusal_abstains():
    store, s = _store()
    ans = matter_ask(store, s, "A", "what is the interest rate",
                     llm=FakeLLM({"answer_possible": False, "claims": []}), nli=_nli)
    assert ans.abstained


def test_garbage_output_abstains():
    store, s = _store()
    for garbage in ["I think the answer is 42.", "{broken json", ""]:
        ans = matter_ask(store, s, "A", "anything", llm=FakeLLM(garbage), nli=_nli)
        assert ans.abstained


def test_nonexistent_excerpt_number_dropped():
    store, s = _store()
    ans = matter_ask(store, s, "A", "leverage ratio the Borrower shall maintain",
                     llm=FakeLLM({"answer_possible": True, "claims": [
                         {"text": "The Borrower shall maintain a leverage ratio below 3.0x",
                          "sources": [99]}]}),                 # invented excerpt
                     nli=_nli)
    assert ans.abstained


# ---- real model: open question on the real agreement ------------------------
def test_real_open_question_on_cenveo():
    pytest.importorskip("transformers")
    from pathlib import Path

    from beevr.ingest import ingest_document
    from beevr.llm import TransformersLLM
    from beevr.models import CrossEncoderNLI
    try:
        llm = TransformersLLM()
        nli = CrossEncoderNLI()
    except Exception as ex:
        pytest.skip(f"models unavailable: {ex}")

    store = Store(audit=AuditLog())
    store.put_matter("C", client="X", name="Cenveo")
    text = (Path(__file__).parent.parent / "test-data" /
            "cenveo-credit-agreement.txt").read_text(encoding="utf-8")
    ingest_document(store, "C", "doc", text.encode(), "cenveo.txt")
    s = Session("alice", matter_grants=frozenset({"C"}))

    ans = matter_ask(store, s, "C",
                     "Who is the Administrative Agent under the Credit Agreement?",
                     llm=llm, nli=nli)
    assert not ans.abstained
    assert "bank of america" in ans.answer_text.lower()
    assert all(c.citations for c in ans.claims)        # every claim cited

    # unanswerable open question must abstain, not guess
    ans2 = matter_ask(store, s, "C", "What is the applicable interest rate margin?",
                      llm=llm, nli=nli)
    assert ans2.abstained
