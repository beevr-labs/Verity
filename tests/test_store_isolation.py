"""Matter isolation tests — FR-KM-08 (🔒), FR-SE-04/06.

Release-gating: TC-207 (Matter A returns nothing from Matter B).
TC-208 (cross-matter attempt denied + logged). TC-504 (no grant -> no access).
TC-505 (ethical wall filters in retrieval, not just UI).
"""
import pytest

from beevr.audit import AuditLog
from beevr.store import AccessDenied, Session, Store


def _store() -> tuple[Store, AuditLog]:
    audit = AuditLog()
    s = Store(audit=audit)
    # Two matters with deliberately overlapping terms (the isolation set).
    s.put_matter("A", client="Bank-X", name="Facility A")
    s.put_matter("B", client="Bank-Y", name="Facility B", ethical_wall_group="wall-1")
    s.put_document("dA", "A", mime="application/pdf")
    s.put_document("dB", "B", mime="application/pdf")
    s.put_chunk("cA", "A", text="leverage ratio below 3.0x")
    s.put_chunk("cB", "B", text="leverage ratio below 3.0x")   # same terms!
    s.put_entity("eA", "A", type="party", value="Acme")
    s.put_entity("eB", "B", type="party", value="Acme")
    s.put_edge("eB", "owes", "eB", "cB", "B")
    return s, audit


ALICE_A = Session("alice", matter_grants=frozenset({"A"}))


@pytest.mark.release_gating
def test_query_in_A_returns_nothing_from_B_TC207():
    s, _ = _store()
    chunks = s.retrieve_chunks(ALICE_A, "A", contains="leverage ratio")
    ids = {c.id for c in chunks}
    assert ids == {"cA"}          # cB has identical text but is never returned
    docs = {d.id for d in s.list_documents(ALICE_A, "A")}
    assert docs == {"dA"}


@pytest.mark.release_gating
def test_cross_matter_access_denied_and_logged_TC208():
    s, audit = _store()
    with pytest.raises(AccessDenied) as ei:
        s.retrieve_chunks(ALICE_A, "B", ts="2026-07-02T10:00:00Z")   # A-user reaches for B
    assert ei.value.matter_id == "B"
    # the denial is recorded in the audit log
    events = [e for e in audit.events if e.type == "access_denied"]
    assert len(events) == 1 and events[0].target == "B"
    assert audit.verify() is True


def test_no_grant_no_access_TC504():
    s, _ = _store()
    nobody = Session("eve", matter_grants=frozenset())   # deny-by-default
    with pytest.raises(AccessDenied):
        s.list_documents(nobody, "A")


def test_candidate_resolution_drops_foreign_chunks():
    s, _ = _store()
    # a buggy/adversarial retriever hands us a B chunk while scoped to A
    allowed = s.resolve_candidates(ALICE_A, "A", ["cA", "cB", "ghost"])
    assert allowed == {"cA"}       # only in-matter candidates survive


def test_graph_expansion_is_matter_scoped():
    s, _ = _store()
    # scoped to A, expanding from an entity id that only has edges in B -> nothing
    neighbors = s.graph_neighbors(ALICE_A, "A", {"eB"})
    assert neighbors == []


def test_ethical_wall_filters_in_retrieval_TC505():
    s, audit = _store()
    # Bob is granted B but is walled off from wall-1 -> retrieval denied+logged.
    bob = Session("bob", matter_grants=frozenset({"B"}),
                  walled_groups=frozenset({"wall-1"}))
    with pytest.raises(AccessDenied) as ei:
        s.retrieve_chunks(bob, "B", ts="2026-07-02T11:00:00Z")
    assert ei.value.reason == "ethical_wall"
    assert any(e.type == "access_denied" for e in audit.events)
