"""UI smoke tests — doc 17 SPA + source-viewer endpoint (S5, FR-UX-01)."""
import pytest
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app


@pytest.fixture()
def api():
    state = AppState()
    client = TestClient(create_app(state))
    tok = client.post("/auth/token", json={
        "sub": "alice", "roles": ["user"], "matter_grants": ["A"]}).json()["token"]
    hdr = {"Authorization": f"Bearer {tok}"}
    client.post("/matters", json={"id": "A", "client": "X", "name": "Facility A"}, headers=hdr)
    state.store.put_chunk("cA", "A", text="The Borrower shall maintain a leverage ratio below 3.0x.",
                          document_id="agr")
    return client, hdr


def test_spa_served_with_trust_affordances(api):
    client, _ = api
    html = client.get("/").text
    # citation-first, abstain, HITL, kill switch, matter switcher (doc 17 §4)
    assert "matter-switcher" in html                    # S2 isolation visible
    assert "Insufficient evidence" in html              # abstain-over-guess
    assert "nothing is saved" in html.lower()           # HITL banner (FR-UX-02)
    assert "Kill switch" in html                        # FR-AG-06 reachable
    assert "How this answer was produced" in html       # defensibility expander


def test_source_viewer_endpoint_returns_chunk(api):
    client, hdr = api
    r = client.get("/matters/A/source/cA", headers=hdr)
    assert r.status_code == 200
    assert "leverage ratio" in r.json()["text"]
    assert r.json()["document_id"] == "agr"


def test_source_viewer_isolation(api):
    client, hdr = api
    # chunk of another matter -> 404 within this matter; no grant -> 403
    r = client.get("/matters/A/source/ghost", headers=hdr)
    assert r.status_code == 404
    tok_b = client.post("/auth/token", json={
        "sub": "mallory", "roles": ["user"], "matter_grants": ["B"]}).json()["token"]
    r = client.get("/matters/A/source/cA", headers={"Authorization": f"Bearer {tok_b}"})
    assert r.status_code == 403
