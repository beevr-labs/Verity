"""Egress control tests — FR-SE-01 (🔒), SEC-1/2, CMP-9.

Release-gating: TC-501 (zero outbound customer-data on the default flow).
Also proves the sanctioned escalation path is the ONLY way out.
"""
import socket

import pytest

from beevr.audit import AuditLog
from beevr.egress import EgressDenied, EgressGuard, capture_egress
from beevr.pipeline import matter_qa
from beevr.store import Session, Store
from beevr.verification import lexical_overlap


def _nli(span: str, claim: str) -> float:
    return lexical_overlap(claim, span)


def _store() -> tuple[Store, Session]:
    store = Store(audit=AuditLog())
    store.put_matter("A", client="X", name="Facility A")
    store.put_chunk("cA1", "A", text="The Borrower shall maintain a leverage ratio below 3.0x.")
    return store, Session("alice", matter_grants=frozenset({"A"}))


@pytest.mark.release_gating
def test_full_flow_makes_zero_egress_TC501():
    store, session = _store()
    with capture_egress() as attempts:
        result = matter_qa(store, session, "A",
                           "leverage ratio the Borrower shall maintain",
                           nli=_nli, ts="2026-07-02T12:00:00Z")
    assert attempts == []                 # 0 outbound customer-data connections
    assert not result.abstained           # and it actually answered (real flow ran)


def test_monitor_catches_a_stray_outbound():
    # Proves the harness is not vacuous: a rogue outbound IS detected/blocked.
    with pytest.raises(EgressDenied):
        with capture_egress(block=True):
            s = socket.socket()
            try:
                s.connect(("8.8.8.8", 53))   # blocked before any real connection
            finally:
                s.close()


# ---- sanctioned escalation path is the only way out ----------------------
def test_default_deny_no_frontier_TC501():
    guard = EgressGuard()  # nothing configured
    with pytest.raises(EgressDenied, match="default-deny"):
        guard.send("https://api.frontier.example/v1", "hello",
                   redacted=True, approved=True)


def test_escalation_allowed_only_when_all_conditions_met():
    guard = EgressGuard(frontier_endpoint="https://api.frontier.example/v1",
                        escalation_enabled=True)
    # missing redaction -> denied
    with pytest.raises(EgressDenied, match="redacted"):
        guard.send("https://api.frontier.example/v1", "x", approved=True)
    # missing approval -> denied
    with pytest.raises(EgressDenied, match="approved"):
        guard.send("https://api.frontier.example/v1", "x", redacted=True)
    # wrong destination -> denied
    with pytest.raises(EgressDenied, match="frontier"):
        guard.send("https://evil.example/v1", "x", redacted=True, approved=True)
    # all conditions met -> allowed, and recorded
    assert guard.send("https://api.frontier.example/v1", "redacted slice",
                      redacted=True, approved=True) == "sent"
    assert guard.sent == [("api.frontier.example", "redacted slice")]


def test_escalation_disabled_by_default_denies():
    guard = EgressGuard(frontier_endpoint="https://api.frontier.example/v1")
    with pytest.raises(EgressDenied, match="disabled"):
        guard.send("https://api.frontier.example/v1", "x", redacted=True, approved=True)
