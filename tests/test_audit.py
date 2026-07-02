"""Audit hash-chain tests — FR-AU-05, TC-402 (tamper-evident, append-only)."""
import pytest

from beevr.audit import AuditLog, TamperError, tamper


def _seed() -> AuditLog:
    log = AuditLog()
    log.append(actor="alice", actor_kind="user", type="query",
               target="matter-1", payload="what obligations?", ts="2026-07-02T10:00:00Z")
    log.append(actor="agent-1", actor_kind="agent", type="extract",
               target="doc-1", payload="{...}", ts="2026-07-02T10:00:05Z")
    log.append(actor="bob", actor_kind="user", type="approve",
               target="proposal-9", payload="approved", ts="2026-07-02T10:01:00Z")
    return log


def test_clean_chain_verifies():
    assert _seed().verify() is True


def test_chain_links_prev_to_this():
    log = _seed()
    evs = log.events
    assert evs[0].prev_hash == "0" * 64
    assert evs[1].prev_hash == evs[0].this_hash
    assert evs[2].prev_hash == evs[1].this_hash


@pytest.mark.release_gating
def test_edit_is_detected_TC402():
    log = _seed()
    # Simulate an attacker editing a stored event's payload hash in place.
    log._events[1] = tamper(log._events[1], payload_hash="deadbeef" * 8)
    with pytest.raises(TamperError) as ei:
        log.verify()
    assert ei.value.seq == 1


@pytest.mark.release_gating
def test_deletion_reorder_is_detected_TC402():
    log = _seed()
    del log._events[1]  # drop the middle event
    with pytest.raises(TamperError):
        log.verify()


def test_crypto_shred_keeps_chain_valid_FR_AU_06():
    log = _seed()
    log.crypto_shred(1)  # content destroyed elsewhere; metadata retained
    assert log.verify() is True
