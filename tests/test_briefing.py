"""Matter briefing tests — proactive deadline radar + action items +
suggested questions. `today` is injected for determinism."""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app
from beevr.audit import AuditLog
from beevr.briefing import build_briefing
from beevr.store import Session, Store

TODAY = date(2026, 7, 3)


def _store():
    s = Store(audit=AuditLog())
    s.put_matter("A", client="X", name="Facility A")
    s.put_chunk("c-past", "A", text="The waiver expires on 31 March 2026 unless renewed.")
    s.put_chunk("c-soon", "A", text="The Borrower shall deliver financials no later than August 15, 2026.")
    s.put_chunk("c-far", "A", text="This Agreement terminates on 2027-12-31.")
    s.put_chunk("c-nodate", "A", text="The Guarantor waives notice of presentment.")
    s.put_chunk("c-nocue", "A", text="The meeting of June 1, 2026 approved the budget.")  # date, no cue
    return s, Session("alice", matter_grants=frozenset({"A"}))


def test_deadline_radar_sorted_and_flagged():
    store, s = _store()
    b = build_briefing(store, s, "A", today=TODAY)
    assert [d.status for d in b.deadlines] == ["overdue", "due_soon", "upcoming"]
    assert b.deadlines[0].due == date(2026, 3, 31)          # sorted by date
    assert b.deadlines[1].days_left == 43                   # Aug 15 from Jul 3
    assert all(d.chunk_id for d in b.deadlines)             # every item cited
    # a date WITHOUT a deadline cue word is not a deadline
    assert not any(d.chunk_id == "c-nocue" for d in b.deadlines)


def test_action_items_reflect_state():
    store, s = _store()
    b = build_briefing(store, s, "A", today=TODAY)
    joined = " ".join(b.actions)
    assert "extraction has not been run" in joined.lower()
    assert "in the past" in joined                          # overdue nudge
    assert "90 days" in joined                              # due-soon nudge


def test_suggested_questions_track_content_signals():
    store, s = _store()
    qs = build_briefing(store, s, "A", today=TODAY).suggested_questions
    joined = " ".join(qs).lower()
    assert "terminate" in joined                            # termination signal
    assert "notice" in joined                               # notice signal
    assert len(qs) <= 8


def test_briefing_isolation():
    store, _ = _store()
    outsider = Session("mallory", matter_grants=frozenset({"B"}))
    with pytest.raises(Exception):
        build_briefing(store, outsider, "A", today=TODAY)


def test_briefing_endpoint_and_pending_hitl():
    state = AppState()
    client = TestClient(create_app(state))
    tok = client.post("/auth/token", json={
        "sub": "alice", "roles": ["user"], "matter_grants": ["A"]}).json()["token"]
    hdr = {"Authorization": f"Bearer {tok}"}
    client.post("/matters", json={"id": "A", "client": "X", "name": "MA"}, headers=hdr)
    state.store.put_chunk("c0", "A",
        text="The Borrower shall maintain a leverage ratio below 3.0x until 31 March 2027, due quarterly.")
    # run extraction -> proposals awaiting HITL
    client.post("/matters/A/agent/runs", json={"workflow": "obligation_extraction"},
                headers=hdr)
    r = client.get("/matters/A/briefing?today=2026-07-03", headers=hdr)
    assert r.status_code == 200
    body = r.json()
    assert any("awaiting your HITL review" in a for a in body["actions"])
    assert any(d["due"] == "2027-03-31" for d in body["deadlines"])
    assert any("covenant" in q.lower() for q in body["suggested_questions"])
    # cross-matter -> 403
    tok_b = client.post("/auth/token", json={
        "sub": "m", "roles": ["user"], "matter_grants": ["B"]}).json()["token"]
    assert client.get("/matters/A/briefing",
                      headers={"Authorization": f"Bearer {tok_b}"}).status_code == 403
