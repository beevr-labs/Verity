"""Egress control — doc 16 §1, FR-SE-01 (🔒 zero egress), SEC-1/2.

Two things live here:

1. `EgressGuard` — the application's ONLY sanctioned outbound path (layer (b)
   of doc 16 §1). Default-deny: with no frontier endpoint configured, nothing
   may leave. The single exception is the hybrid-escalation path, and only when
   ALL hold: escalation enabled, destination == the configured frontier, payload
   redacted, and human-approved (SEC-2, CMP-9).

2. `capture_egress()` — the proof monitor (TASK-081). It instruments socket
   connections so a test/harness can assert that a full pipeline run makes ZERO
   outbound customer-data connections (TC-501). `block=True` also enforces it.

The network-namespace + packet-sniffer (layer (a)) is the CI complement on Linux
(see scripts/egress_proof.py); this module proves the app layer offline.
"""
from __future__ import annotations

import contextlib
import ipaddress
import socket
from urllib.parse import urlparse


class EgressDenied(Exception):
    """Raised when an outbound connection is attempted outside the sanctioned path."""


# --------------------------------------------------------------------------
# 1. Sanctioned-path guard
# --------------------------------------------------------------------------
class EgressGuard:
    def __init__(self, frontier_endpoint: str | None = None,
                 escalation_enabled: bool = False):
        # Default: zero egress. Both must be set to allow the escalation path.
        self.frontier_endpoint = frontier_endpoint
        self.escalation_enabled = escalation_enabled
        self.sent: list[tuple[str, str]] = []  # (host, payload) audit of what left

    @property
    def _frontier_host(self) -> str | None:
        if not self.frontier_endpoint:
            return None
        return urlparse(self.frontier_endpoint).hostname

    def send(self, url: str, payload: str, *, redacted: bool = False,
             approved: bool = False) -> str:
        """The only way anything leaves the boundary. Denies unless every
        condition of the sanctioned escalation path is met."""
        if not self.frontier_endpoint:                       # FR-SE-01 / TC-501
            raise EgressDenied("default-deny: no frontier endpoint configured")
        if not self.escalation_enabled:
            raise EgressDenied("escalation disabled")
        if urlparse(url).hostname != self._frontier_host:    # SEC-2
            raise EgressDenied("destination is not the configured frontier endpoint")
        if not redacted:
            raise EgressDenied("payload not redacted")
        if not approved:                                     # CMP-9
            raise EgressDenied("escalation not human-approved")
        self.sent.append((self._frontier_host, payload))
        return "sent"


# --------------------------------------------------------------------------
# 2. Proof monitor — instruments outbound sockets
# --------------------------------------------------------------------------
def _is_remote(address) -> bool:
    """True if `address` leaves the host/boundary (not loopback/localhost)."""
    if not isinstance(address, tuple) or not address:
        return False  # AF_UNIX etc. — in-process
    host = address[0]
    if host in ("localhost", ""):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_loopback or ip.is_link_local)
    except ValueError:
        return True   # an unresolved hostname => a remote destination


@contextlib.contextmanager
def capture_egress(block: bool = False):
    """Record (and optionally block) every outbound socket connection.

    Yields a list that fills with remote destinations attempted during the
    context. `block=True` raises EgressDenied on the first remote attempt.
    """
    attempts: list = []
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _guarded(orig):
        def wrapper(self, address, *a, **k):
            if _is_remote(address):
                attempts.append(address)
                if block:
                    raise EgressDenied(f"blocked outbound connection to {address!r}")
            return orig(self, address, *a, **k)
        return wrapper

    socket.socket.connect = _guarded(orig_connect)
    socket.socket.connect_ex = _guarded(orig_connect_ex)
    try:
        yield attempts
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
