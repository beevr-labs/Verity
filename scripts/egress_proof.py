#!/usr/bin/env python
"""Egress proof harness — TASK-081, FR-SE-01, TC-501 (release-gating).

Runs the reference matter-Q&A pipeline (ingest already done in-memory) under a
socket monitor and asserts ZERO outbound customer-data connections. Exits
non-zero if anything tries to leave the boundary.

This is the application-layer (doc 16 §1 layer (b)) proof and runs anywhere,
including Windows/CI without Docker. The network-layer proof (layer (a)) wraps
the SAME flow in a Linux network namespace with a packet sniffer:

    sudo unshare --net -- bash -c '
        ip link set lo up
        python scripts/egress_proof.py            # loopback only; no default route
    '
    # with no default route configured, any real egress attempt fails outright,
    # and this harness additionally proves no attempt is even made.

Usage:  python scripts/egress_proof.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beevr.audit import AuditLog
from beevr.egress import capture_egress
from beevr.store import Session, Store
from beevr.verification import lexical_overlap


def _stub_nli(span: str, claim: str) -> float:
    return lexical_overlap(claim, span)


def _seed_store() -> tuple[Store, Session]:
    audit = AuditLog()
    store = Store(audit=audit)
    store.put_matter("A", client="Bank-X", name="Facility A")
    store.put_document("dA", "A", mime="application/pdf")
    store.put_chunk("cA1", "A", text="The Borrower shall maintain a leverage ratio below 3.0x.")
    store.put_chunk("cA2", "A", text="This Agreement terminates on 31 March 2027.")
    session = Session("alice", matter_grants=frozenset({"A"}))
    return store, session


def main() -> int:
    from beevr.pipeline import matter_qa

    store, session = _seed_store()

    with capture_egress() as attempts:
        result = matter_qa(store, session, "A",
                           "The Borrower shall maintain a leverage ratio below 3.0x",
                           nli=_stub_nli, ts="2026-07-02T12:00:00Z")

    leaked = len(attempts)
    print(f"pipeline: abstained={result.abstained} "
          f"claims={len(result.claims)} confidence={result.confidence}")
    print(f"audit events emitted: {len(store.audit.events)} (chain ok: {store.audit.verify()})")
    print(f"outbound customer-data connections attempted: {leaked}")

    if leaked:
        print("EGRESS PROOF: FAIL --", attempts)
        return 1
    print("EGRESS PROOF: PASS -- 0 outbound bytes (TC-501)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
