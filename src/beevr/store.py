"""Data-access layer with matter isolation — doc 12 §3, FR-KM-08 (🔒), FR-SE-04.

This is the PRIMARY isolation control (Postgres RLS in db/migrations/0001_init.sql
is defense in depth behind it). The contract, enforced here and nowhere else
("never in ad-hoc queries"):

  * deny-by-default: a session may only touch matters in its `matter_grants`
  * every read is filtered to a single scoped `matter_id`
  * a cross-matter access attempt is denied AND logged (TC-208)
  * results scoped to matter A never contain matter B rows (TC-207)
  * ethical walls filter walled matters out of reads, not just the UI (FR-SE-06)

All reads go through `_authorize(...)`; there is no other query path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLog


@dataclass(frozen=True)
class Session:
    user_id: str
    role: str = "user"
    matter_grants: frozenset[str] = field(default_factory=frozenset)
    walled_groups: frozenset[str] = field(default_factory=frozenset)  # ethical walls


class AccessDenied(Exception):
    def __init__(self, matter_id: str, reason: str):
        self.matter_id = matter_id
        self.reason = reason
        super().__init__(f"access denied to matter {matter_id!r}: {reason}")


@dataclass
class _Row:
    id: str
    matter_id: str
    data: dict = field(default_factory=dict)


class Store:
    """In-memory model of the matter-scoped tables (doc 12 §2)."""

    def __init__(self, audit: AuditLog | None = None):
        self.audit = audit
        self.matters: dict[str, dict] = {}
        self.documents: dict[str, _Row] = {}
        self.chunks: dict[str, _Row] = {}
        self.entities: dict[str, _Row] = {}
        self.edges: list[_Row] = []
        self.answers: dict[str, _Row] = {}
        self.citations: dict[str, _Row] = {}

    # ---- writes (matter_id is intrinsic to every row) --------------------
    def put_matter(self, id: str, client: str, name: str,
                   ethical_wall_group: str | None = None) -> None:
        self.matters[id] = {"client": client, "name": name,
                            "ethical_wall_group": ethical_wall_group}

    def put_document(self, id: str, matter_id: str, **data) -> None:
        self.documents[id] = _Row(id, matter_id, data)

    def put_chunk(self, id: str, matter_id: str, text: str, **data) -> None:
        self.chunks[id] = _Row(id, matter_id, {"text": text, **data})

    def put_entity(self, id: str, matter_id: str, type: str, value: str) -> None:
        self.entities[id] = _Row(id, matter_id, {"type": type, "value": value})

    def put_edge(self, from_entity: str, rel: str, to_entity: str,
                 source_chunk_id: str, matter_id: str) -> None:
        self.edges.append(_Row(f"{from_entity}->{to_entity}", matter_id, {
            "from": from_entity, "rel": rel, "to": to_entity,
            "source_chunk_id": source_chunk_id}))

    # ---- authorization (the single choke point) --------------------------
    def _authorize(self, session: Session, matter_id: str, action: str,
                   ts: str = "") -> None:
        matter = self.matters.get(matter_id)
        walled = (matter is not None
                  and matter["ethical_wall_group"] in session.walled_groups)
        if matter_id not in session.matter_grants or walled:
            reason = "ethical_wall" if walled else "no_matter_grant"
            if self.audit is not None:                       # TC-208: log the denial
                self.audit.append(actor=session.user_id, actor_kind="user",
                                  type="access_denied", target=matter_id,
                                  payload=f"{action}:{reason}", ts=ts)
            raise AccessDenied(matter_id, reason)

    def _scoped(self, rows, matter_id: str):
        # defense in depth: filter to the scoped matter even after authorize.
        return [r for r in rows if r.matter_id == matter_id]

    # ---- reads (all scoped to one authorized matter) ---------------------
    def list_matters(self, session: Session) -> list[dict]:
        """Only matters the session is granted and not walled off from."""
        out = []
        for mid, m in self.matters.items():
            if mid in session.matter_grants and \
                    m["ethical_wall_group"] not in session.walled_groups:
                out.append({"id": mid, **m})
        return out

    def list_documents(self, session: Session, matter_id: str, ts: str = "") -> list[_Row]:
        self._authorize(session, matter_id, "list_documents", ts)
        return self._scoped(self.documents.values(), matter_id)

    def retrieve_chunks(self, session: Session, matter_id: str,
                        contains: str | None = None, ts: str = "") -> list[_Row]:
        self._authorize(session, matter_id, "retrieve_chunks", ts)
        rows = self._scoped(self.chunks.values(), matter_id)
        if contains is not None:
            rows = [r for r in rows if contains.lower() in r.data["text"].lower()]
        return rows

    def resolve_candidates(self, session: Session, matter_id: str,
                           chunk_ids: list[str], ts: str = "") -> set[str]:
        """Filter retriever/graph candidate ids down to the scoped matter — the
        `allowed` set handed to fusion (defense in depth, doc 22 §2.1)."""
        self._authorize(session, matter_id, "resolve_candidates", ts)
        return {cid for cid in chunk_ids
                if cid in self.chunks and self.chunks[cid].matter_id == matter_id}

    def graph_neighbors(self, session: Session, matter_id: str,
                        entity_ids: set[str], ts: str = "") -> list[_Row]:
        self._authorize(session, matter_id, "graph_neighbors", ts)
        return [e for e in self._scoped(self.edges, matter_id)
                if e.data["from"] in entity_ids]
