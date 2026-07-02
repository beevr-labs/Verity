"""LLM layer tests — doc 14/15 §6.

Offline: prompt discipline + defensive JSON parsing (fake provider).
Real: Qwen2.5-1.5B-Instruct extraction on GPU + the full agent pipeline with
the real LLM extractor — items still gated by NLI verification. Skips when
weights/runtime are absent.
"""
import json

import pytest

from beevr.llm import PROMPT, LlmExtractor
from beevr.locator import Locator

LOC = Locator("c0", "pdf", char_range=(0, 60), page=1)
CLAUSE = "The Borrower shall maintain a leverage ratio below 3.0x at all times."


# ---- offline: parsing discipline -------------------------------------------
class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, prompt: str, *, max_new_tokens: int = 512) -> str:
        assert "Output ONLY a JSON array" in prompt      # discipline present
        return self.reply


def test_parses_valid_json_array():
    reply = json.dumps([{"item_type": "covenant",
                         "text": "Borrower must maintain leverage ratio below 3.0x",
                         "party": "Borrower", "counterparty": "",
                         "trigger_or_due": ""}])
    items = LlmExtractor(FakeLLM(reply)).extract(CLAUSE, LOC)
    assert len(items) == 1 and items[0].item_type == "covenant"
    assert items[0].locator is LOC                       # citation pinned to source


def test_rejects_prose_invalid_type_and_garbage():
    assert LlmExtractor(FakeLLM("The clause creates a covenant.")).extract(CLAUSE, LOC) == []
    assert LlmExtractor(FakeLLM('[{"item_type": "world_peace", "text": "x"}]')).extract(CLAUSE, LOC) == []
    assert LlmExtractor(FakeLLM("[{broken json")).extract(CLAUSE, LOC) == []
    assert LlmExtractor(FakeLLM("[]")).extract(CLAUSE, LOC) == []


def test_json_embedded_in_chatter_is_recovered():
    reply = 'Sure! Here is the extraction:\n[{"item_type": "obligation", "text": "Borrower shall pay"}]\nHope this helps!'
    items = LlmExtractor(FakeLLM(reply)).extract(CLAUSE, LOC)
    assert len(items) == 1 and items[0].text == "Borrower shall pay"


# ---- real model (skip without runtime/weights) ------------------------------
@pytest.fixture(scope="module")
def llm():
    pytest.importorskip("transformers")
    from beevr.llm import TransformersLLM
    try:
        return TransformersLLM()
    except Exception as ex:
        pytest.skip(f"Qwen weights unavailable: {ex}")


def test_real_llm_extracts_covenant(llm):
    items = LlmExtractor(llm).extract(CLAUSE, LOC)
    assert len(items) >= 1
    joined = " ".join(i.text.lower() for i in items)
    assert "leverage ratio" in joined and "3.0x" in joined
    assert items[0].item_type in ("covenant", "obligation")


def test_real_llm_returns_empty_for_no_obligation(llm):
    items = LlmExtractor(llm).extract(
        "The parties enjoyed a pleasant lunch in Geneva.", LOC)
    assert items == []


def test_real_llm_agent_pipeline_end_to_end(llm):
    """The full doc-15 §5 pipeline with a REAL LLM extractor: extract -> NLI
    verify -> HITL -> Kernel persist. Hallucinations die at the verify gate."""
    from beevr.agent import AgentRun
    from beevr.audit import AuditLog
    from beevr.models import CrossEncoderNLI
    from beevr.store import Session, Store

    try:
        nli = CrossEncoderNLI()
    except Exception as ex:
        pytest.skip(f"NLI unavailable: {ex}")

    store = Store(audit=AuditLog())
    store.put_matter("A", client="X", name="Facility A")
    store.put_chunk("c0", "A", text=CLAUSE, document_id="agr")
    store.put_chunk("c1", "A", text="This Agreement terminates on 31 March 2027.",
                    document_id="agr")

    run = AgentRun(run_id="r1", matter_id="A",
                   session=Session("alice", matter_grants=frozenset({"A"})),
                   store=store, nli=nli, extractor=LlmExtractor(llm))
    items = run.extract_phase()
    assert len(items) >= 1
    assert all(i.verified for i in items)                # every survivor is grounded
    run.decide(0, "approve", idempotency_key="k0")
    assert run.persisted and run.persisted[0]["citation"]["verified"] is True
    assert store.audit.verify() is True
