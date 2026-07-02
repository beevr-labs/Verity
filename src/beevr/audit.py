"""Tamper-evident audit log — doc 16 §3, FR-AU-05 (resolves OI-4).

Append-only, hash-chained:
    this_hash = H(prev_hash ‖ actor ‖ actor_kind ‖ type ‖ target ‖ payload_hash ‖ ts)

Payloads are *hashed*, not stored verbatim — so content deletion-on-request
(crypto-shred, FR-SE-08/FR-AU-06) leaves the chain verifiable. Any edit or
deletion of a recorded event breaks the chain and is detectable (TC-402).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

GENESIS = "0" * 64


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def payload_hash(payload: str) -> str:
    """Hash of the event payload. The raw payload never has to be stored."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    actor: str
    actor_kind: str  # "user" | "agent"
    type: str
    target: str
    payload_hash: str
    ts: str
    prev_hash: str
    this_hash: str

    def recompute(self) -> str:
        return _h(self.prev_hash, self.actor, self.actor_kind, self.type,
                  self.target, self.payload_hash, self.ts)


class TamperError(Exception):
    def __init__(self, seq: int, reason: str):
        self.seq = seq
        super().__init__(f"audit chain broken at seq={seq}: {reason}")


class AuditLog:
    """In-memory append-only hash-chained log (the DB enforces the same via
    REVOKE UPDATE/DELETE + BIGSERIAL seq; see doc 12 §2)."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    @property
    def head(self) -> str:
        return self._events[-1].this_hash if self._events else GENESIS

    def append(self, *, actor: str, actor_kind: str, type: str, target: str,
               payload: str, ts: str) -> AuditEvent:
        seq = len(self._events)
        prev = self.head
        ph = payload_hash(payload)
        this = _h(prev, actor, actor_kind, type, target, ph, ts)
        ev = AuditEvent(seq, actor, actor_kind, type, target, ph, ts, prev, this)
        self._events.append(ev)
        return ev

    def crypto_shred(self, seq: int) -> None:
        """Deletion-on-request: the referenced content is destroyed elsewhere;
        the audit metadata (and its payload_hash) stays, so the chain still
        verifies (FR-AU-06). This does NOT mutate hashed fields."""
        # No-op on the chain by design — retained here to document the contract.
        if not (0 <= seq < len(self._events)):
            raise IndexError(seq)

    def verify(self) -> bool:
        """Recompute the whole chain; raise TamperError at the first break."""
        prev = GENESIS
        for i, ev in enumerate(self._events):
            if ev.seq != i:
                raise TamperError(ev.seq, "sequence gap/reorder")
            if ev.prev_hash != prev:
                raise TamperError(ev.seq, "prev_hash mismatch (event removed/reordered)")
            if ev.recompute() != ev.this_hash:
                raise TamperError(ev.seq, "content altered (this_hash mismatch)")
            prev = ev.this_hash
        return True


def tamper(ev: AuditEvent, **changes) -> AuditEvent:
    """Test helper: produce an illicitly edited copy of a stored event."""
    return replace(ev, **changes)
