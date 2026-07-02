"""Playbook review tests — US-303b, FR-AG-08/09 (TC-306/307).

Offline (FakeLLM): grounding discipline — an invented quote is discarded;
flags cite real clauses; redlines are proposals only. Real (Qwen): the
guaranty governed by OHIO law must be flagged as a DEVIATION from a
New-York-law playbook, with a grounded quote.
"""
import json

import pytest

from beevr.audit import AuditLog
from beevr.playbook import Rule, review
from beevr.store import Session, Store

NY_RULE = Rule("GL-1", "governing law",
               "The agreement must be governed by the laws of the State of New York",
               "high", "governed by laws")


def _store(text: str):
    s = Store(audit=AuditLog())
    s.put_matter("A", client="X", name="M")
    s.put_chunk("c0", "A", text=text)
    return s, Session("alice", matter_grants=frozenset({"A"}))


class FakeLLM:
    def __init__(self, judge_reply: dict | str, redline: str = "Redline text."):
        self.judge_reply = judge_reply if isinstance(judge_reply, str) \
            else json.dumps(judge_reply)
        self.redline = redline

    def generate(self, prompt, *, max_new_tokens=300):
        if "PLAYBOOK RULE" in prompt and "CLAUSE QUOTE" in prompt:
            return self.redline
        return self.judge_reply


OHIO_CLAUSE = ("The provisions of this Agreement shall be governed by and "
               "construed in accordance with Ohio law.")


def test_deviation_flag_cites_real_clause_TC306():
    store, s = _store(OHIO_CLAUSE)
    llm = FakeLLM({"relevant": True, "complies": False,
                   "quote": "governed by and construed in accordance with Ohio law",
                   "rationale": "Ohio, not New York."})
    flags = review(store, s, "A", llm=llm, rules=[NY_RULE])
    f = flags[0]
    assert f.status == "deviates"
    assert f.chunk_id == "c0"                       # cites the real clause
    assert f.quote in OHIO_CLAUSE                   # grounded verbatim
    assert f.suggested_redline == "Redline text."   # proposal present...
    # ...and NOTHING was modified: the stored clause is untouched (TC-307)
    assert store.chunks["c0"].data["text"] == OHIO_CLAUSE


def test_invented_quote_is_discarded():
    store, s = _store(OHIO_CLAUSE)
    llm = FakeLLM({"relevant": True, "complies": False,
                   "quote": "governed by the laws of Delaware",   # NOT in clause
                   "rationale": "made up"})
    flags = review(store, s, "A", llm=llm, rules=[NY_RULE])
    assert flags[0].status == "absent"              # fabricated evidence -> no flag


def test_complies_flag():
    text = "This Agreement shall be governed by the laws of the State of New York."
    store, s = _store(text)
    llm = FakeLLM({"relevant": True, "complies": True,
                   "quote": "governed by the laws of the State of New York",
                   "rationale": "Matches the standard."})
    flags = review(store, s, "A", llm=llm, rules=[NY_RULE])
    assert flags[0].status == "complies" and flags[0].suggested_redline == ""


def test_absent_when_no_relevant_clause():
    store, s = _store("The parties enjoyed lunch in Geneva.")
    llm = FakeLLM({"relevant": False, "complies": False, "quote": "", "rationale": ""})
    flags = review(store, s, "A", llm=llm, rules=[NY_RULE])
    assert flags[0].status == "absent"
    assert "No relevant clause" in flags[0].rationale


def test_audit_trail_per_rule():
    store, s = _store(OHIO_CLAUSE)
    llm = FakeLLM({"relevant": False, "complies": False, "quote": "", "rationale": ""})
    review(store, s, "A", llm=llm, rules=[NY_RULE])
    assert any(e.type == "playbook_check" and e.target == "GL-1"
               for e in store.audit.events)


def test_isolation():
    store, _ = _store(OHIO_CLAUSE)
    outsider = Session("mallory", matter_grants=frozenset({"B"}))
    with pytest.raises(Exception):
        review(store, outsider, "A", llm=FakeLLM("{}"), rules=[NY_RULE])


# ---- real model: the Ohio guaranty vs a New-York playbook -------------------
def test_real_ohio_guaranty_flagged_as_deviation():
    pytest.importorskip("transformers")
    from pathlib import Path

    from beevr.ingest import ingest_document
    from beevr.llm import TransformersLLM
    try:
        llm = TransformersLLM("Qwen/Qwen2.5-3B-Instruct")
    except Exception as ex:
        pytest.skip(f"model unavailable: {ex}")

    store = Store(audit=AuditLog())
    store.put_matter("G", client="X", name="Guaranty")
    text = (Path(__file__).parent.parent / "test-data" / "guaranty.txt"
            ).read_text(encoding="utf-8")
    ingest_document(store, "G", "doc", text.encode(), "guaranty.txt")
    s = Session("counsel", matter_grants=frozenset({"G"}))

    flags = review(store, s, "G", llm=llm, rules=[NY_RULE])
    f = flags[0]
    assert f.status == "deviates"                    # Ohio law != NY playbook
    assert "ohio" in f.quote.lower()
    assert f.chunk_id                                # cited to a real chunk
    assert f.quote in store.chunks[f.chunk_id].data["text"]   # grounded
    assert f.suggested_redline                       # proposal drafted
