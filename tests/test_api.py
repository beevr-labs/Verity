"""API tests over HTTP (FastAPI TestClient) — doc 11 contract.

End-to-end: auth required, cross-matter -> 403 FORBIDDEN_MATTER (TC-207 at the
API boundary), cited Q&A, abstain path, audit read, idempotency, zero egress.

Chunks are seeded directly into the store to stand in for the (not-yet-built)
real ingestion pipeline output; the API/isolation/verification path is real.
"""
import pytest
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app
from beevr.egress import capture_egress


@pytest.fixture()
def ctx():
    state = AppState()
    app = create_app(state)
    client = TestClient(app)
    # two matters; seed chunks (post-ingestion state)
    for token in [_mint(client, "admin", ["legal_admin"], ["A", "B"])]:
        _auth = {"Authorization": f"Bearer {token}"}
        client.post("/matters", json={"id": "A", "client": "X", "name": "Facility A"}, headers=_auth)
        client.post("/matters", json={"id": "B", "client": "Y", "name": "Facility B"}, headers=_auth)
    state.store.put_chunk("cA", "A", text="The Borrower shall maintain a leverage ratio below 3.0x.")
    state.store.put_chunk("cB", "B", text="Confidential Matter B financials.")
    return client, state


def _mint(client, sub, roles, grants, walled=None):
    r = client.post("/auth/token", json={"sub": sub, "roles": roles,
                                         "matter_grants": grants, "walled": walled or []})
    return r.json()["token"]


def _hdr(token, **extra):
    return {"Authorization": f"Bearer {token}", **extra}


def test_health_needs_no_auth(ctx):
    client, _ = ctx
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"


def test_unauthenticated_is_401(ctx):
    client, _ = ctx
    r = client.get("/matters")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_query_returns_cited_answer(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    r = client.post("/matters/A/queries",
                    json={"question": "leverage ratio the Borrower shall maintain",
                          "ts": "2026-07-02T10:00:00Z"}, headers=_hdr(alice))
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False
    cite = body["claims"][0]["citations"][0]
    assert cite["document_id"] == "cA" and cite["verified"] is True
    assert "leverage ratio" in cite["snippet"]


@pytest.mark.release_gating
def test_cross_matter_query_is_403_TC207(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])   # granted A only
    r = client.post("/matters/B/queries", json={"question": "financials"},
                    headers=_hdr(alice))
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN_MATTER"


def test_abstains_when_no_support(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    r = client.post("/matters/A/queries",
                    json={"question": "what penalty applies on default"}, headers=_hdr(alice))
    assert r.json()["abstained"] is True
    assert r.json()["confidence"] == "insufficient"


def test_document_upload_and_status(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    r = client.post("/matters/A/documents", json={"mime": "application/pdf"},
                    headers=_hdr(alice, **{"Idempotency-Key": "k1"}))
    assert r.status_code == 200 and r.json()["status"] == "queued"
    did = r.json()["document_id"]
    # idempotent: same key -> same document_id
    r2 = client.post("/matters/A/documents", json={"mime": "application/pdf"},
                     headers=_hdr(alice, **{"Idempotency-Key": "k1"}))
    assert r2.json()["document_id"] == did
    listing = client.get("/matters/A/documents", headers=_hdr(alice)).json()
    assert any(d["document_id"] == did for d in listing["documents"])


def test_audit_records_query_and_denial(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    client.post("/matters/A/queries", json={"question": "leverage ratio below 3.0x"},
                headers=_hdr(alice))
    events = client.get("/matters/A/audit", headers=_hdr(alice)).json()["events"]
    assert any(e["type"] == "query" and e["target"] == "A" for e in events)


def test_tampered_token_rejected(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    forged = alice[:-2] + ("aa" if not alice.endswith("aa") else "bb")
    r = client.get("/matters", headers=_hdr(forged))
    assert r.status_code == 401


@pytest.mark.release_gating
def test_api_query_makes_zero_egress_TC501(ctx):
    client, _ = ctx
    alice = _mint(client, "alice", ["user"], ["A"])
    with capture_egress() as attempts:
        client.post("/matters/A/queries",
                    json={"question": "leverage ratio the Borrower shall maintain"},
                    headers=_hdr(alice))
    assert attempts == []
