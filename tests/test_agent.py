"""Obligation-extraction agent tests — FR-AG-07, US-303a.

TC-309 (reference workflow): every item cited to a real clause (AC1); nothing
saved until HITL approval (AC2); no item cites a non-existent clause (AC3);
end-to-end through Kernel + audit (AC4). Plus TC-305 (edit/reject) and TC-308
(kill switch) over HTTP.
"""
import pytest
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app

AGREEMENT = [
    "The Borrower shall maintain a leverage ratio below 3.0x at all times.",
    "This Agreement terminates on 31 March 2027 unless renewed in writing.",
    "The Guarantor agrees to notify the Lender of any default within 5 days.",
    "The parties acknowledge the beautiful weather in Geneva.",   # no obligation
]


@pytest.fixture()
def api():
    state = AppState()
    client = TestClient(create_app(state))
    tok = client.post("/auth/token", json={
        "sub": "alice", "roles": ["user"], "matter_grants": ["A"]}).json()["token"]
    adm = client.post("/auth/token", json={
        "sub": "root", "roles": ["legal_admin"], "matter_grants": ["A"]}).json()["token"]
    hdr = {"Authorization": f"Bearer {tok}"}
    client.post("/matters", json={"id": "A", "client": "X", "name": "Facility A"}, headers=hdr)
    for i, text in enumerate(AGREEMENT):
        state.store.put_chunk(f"c{i}", "A", text=text, document_id="agr-1")
    return client, hdr, {"Authorization": f"Bearer {adm}"}, state


def _start(client, hdr):
    r = client.post("/matters/A/agent/runs",
                    json={"workflow": "obligation_extraction"}, headers=hdr)
    assert r.status_code == 200
    return r.json()


@pytest.mark.release_gating
def test_obligation_extraction_end_to_end_TC309(api):
    client, hdr, _, state = api
    run = _start(client, hdr)
    proposals = run["proposals"]

    # items were extracted from obligation-bearing clauses only
    assert len(proposals) == 3                        # weather sentence excluded
    types = {p["item_type"] for p in proposals}
    assert "covenant" in types and "termination_trigger" in types

    # AC1 + AC3: every proposal is verified and cites a real chunk
    assert all(p["verified"] for p in proposals)
    assert all(p["citation"]["document_id"].startswith("c") for p in proposals)

    # AC2: nothing persisted before approval
    detail = client.get(f"/agent/runs/{run['run_id']}", headers=hdr).json()
    assert detail["persisted"] == []

    # approve item 0 -> persisted exactly once (idempotent)
    for _ in range(2):
        r = client.post(f"/agent/proposals/{run['run_id']}/0/decision",
                        json={"decision": "approve"},
                        headers={**hdr, "Idempotency-Key": "k-0"})
        assert r.json()["status"] == "persisted"
    detail = client.get(f"/agent/runs/{run['run_id']}", headers=hdr).json()
    assert len(detail["persisted"]) == 1              # FR-AG-05 exactly-once

    # AC4: audit trail has the agent extraction + the human approval
    types = [e.type for e in state.store.audit.events]
    assert "extract_obligations" in types and "approve_extraction" in types
    assert state.store.audit.verify() is True


def test_reject_and_edit_TC305(api):
    client, hdr, _, _ = api
    run = _start(client, hdr)
    rid = run["run_id"]
    # reject -> nothing persisted
    r = client.post(f"/agent/proposals/{rid}/1/decision",
                    json={"decision": "reject"}, headers=hdr)
    assert r.json()["status"] == "rejected"
    assert client.get(f"/agent/runs/{rid}", headers=hdr).json()["persisted"] == []
    # edit -> the edited text is what persists
    r = client.post(f"/agent/proposals/{rid}/2/decision",
                    json={"decision": "edit",
                          "edited_args": {"text": "Notify Lender within 10 days."}},
                    headers={**hdr, "Idempotency-Key": "k-2"})
    assert r.json()["status"] == "persisted"
    persisted = client.get(f"/agent/runs/{rid}", headers=hdr).json()["persisted"]
    assert persisted[0]["text"] == "Notify Lender within 10 days."


@pytest.mark.release_gating
def test_kill_switch_blocks_persist_TC308(api):
    client, hdr, _, _ = api
    run = _start(client, hdr)
    rid = run["run_id"]
    client.post(f"/agent/runs/{rid}/kill", headers=hdr)
    r = client.post(f"/agent/proposals/{rid}/0/decision",
                    json={"decision": "approve"}, headers=hdr)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "KILL_SWITCH_ACTIVE"
    assert client.get(f"/agent/runs/{rid}", headers=hdr).json()["persisted"] == []


def test_global_kill_all_requires_admin_and_halts(api):
    client, hdr, adm, _ = api
    run = _start(client, hdr)
    # non-admin cannot kill-all
    assert client.post("/admin/agents/kill-all", headers=hdr).status_code == 403
    # admin can; run is halted and new runs start killed
    assert client.post("/admin/agents/kill-all", headers=adm).status_code == 200
    detail = client.get(f"/agent/runs/{run['run_id']}", headers=hdr).json()
    assert detail["status"] == "killed"
    r2 = client.post("/matters/A/agent/runs",
                     json={"workflow": "obligation_extraction"}, headers=hdr)
    assert r2.json()["status"] == "killed" and r2.json()["proposals"] == []


def test_cross_matter_agent_run_forbidden(api):
    client, _, _, _ = api
    tok_b = client.post("/auth/token", json={
        "sub": "mallory", "roles": ["user"], "matter_grants": ["B"]}).json()["token"]
    r = client.post("/matters/A/agent/runs",
                    json={"workflow": "obligation_extraction"},
                    headers={"Authorization": f"Bearer {tok_b}"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN_MATTER"


def test_unknown_workflow_is_422(api):
    client, hdr, _, _ = api
    r = client.post("/matters/A/agent/runs",
                    json={"workflow": "world_domination"}, headers=hdr)
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION"
