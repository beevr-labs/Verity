"""Obligation & covenant extraction — the P0 reference agent workflow.

doc 15 §5 pipeline, every step traced:
  1. select scope   — documents in the matter (RBAC + isolation via Store)
  2. retrieve spans — the matter's chunks for those documents
  3. extract        — pluggable Extractor produces items with source_locator
                      (read-only, proposals only; FR-AG-07)
  4. verify (SF)    — each item runs the citation-verification pass (doc 13):
                      the extracted text must be entailed by its source span,
                      or it is DROPPED (US-303a AC3 — no fabricated clauses)
  5. HITL           — user approves/edits/rejects per item; NOTHING is saved
                      until approval (AC2)
  6. persist        — `write_extraction_record` through the Kernel
                      (consequential -> checkpoint), idempotent, audited (AC4)

The Extractor is injected: `RuleExtractor` (deterministic legal-pattern rules)
runs offline; a strong-LLM extractor (doc 14 router -> strong) plugs in without
changing this pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .kernel import ActionProposal, Decision, Kernel, KernelReject, Policy
from .locator import Locator
from .store import Session, Store
from .verification import NLI, Verifier

DEFAULT_POLICY = Policy.from_config({
    "actions": [
        {"name": "read_documents", "consequential": False},
        {"name": "extract_obligations", "consequential": False},
        {"name": "write_extraction_record", "consequential": True, "checkpoint": True},
        {"name": "export_report", "consequential": True, "checkpoint": True},
    ],
})


# --------------------------------------------------------------------------
# Extracted item (doc 15 §5 output schema)
# --------------------------------------------------------------------------
@dataclass
class Item:
    item_type: str            # covenant|obligation|date|termination_trigger|regulatory_clause
    text: str
    party: str = ""
    counterparty: str = ""
    trigger_or_due: str = ""
    locator: Locator | None = None
    verified: bool = False
    status: str = "proposed"  # proposed|approved|edited|rejected|persisted


class Extractor(Protocol):
    def extract(self, chunk_text: str, locator: Locator) -> list[Item]:
        ...


# --------------------------------------------------------------------------
# Rule-based extractor (deterministic, offline; LLM extractor plugs in later)
# --------------------------------------------------------------------------
_OBLIG = re.compile(r"\b(shall|must|agrees? to|is required to|undertakes to)\b", re.I)
_COVEN = re.compile(r"\b(maintain|not\s+(?:incur|dispose|permit)|ratio|covenant)\b", re.I)
_TERM = re.compile(r"\b(terminat\w+|renew\w+|expir\w+)\b", re.I)
_DATE = re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
                   r"|\d{4}-\d{2}-\d{2})\b", re.I)
_PARTY = re.compile(r"\b(Borrower|Lender|Guarantor|Company|Supplier|Customer|Tenant|Landlord)\b")


class RuleExtractor:
    """Legal-pattern rules: modal-verb obligations, covenant markers,
    termination/renewal triggers, and dates. Deterministic."""

    def extract(self, chunk_text: str, locator: Locator) -> list[Item]:
        text = chunk_text.strip()
        if not text or not _OBLIG.search(text) and not _TERM.search(text):
            return []
        if _TERM.search(text):
            item_type = "termination_trigger"
        elif _COVEN.search(text):
            item_type = "covenant"
        else:
            item_type = "obligation"
        parties = _PARTY.findall(text)
        date = _DATE.search(text)
        return [Item(item_type=item_type, text=text,
                     party=parties[0] if parties else "",
                     counterparty=parties[1] if len(parties) > 1 else "",
                     trigger_or_due=date.group(0) if date else "",
                     locator=locator)]


# --------------------------------------------------------------------------
# The governed run
# --------------------------------------------------------------------------
@dataclass
class AgentRun:
    run_id: str
    matter_id: str
    session: Session
    store: Store
    nli: NLI
    extractor: Extractor = field(default_factory=RuleExtractor)
    policy: Policy = field(default_factory=lambda: DEFAULT_POLICY)
    status: str = "created"       # created|extracted|done|killed
    items: list[Item] = field(default_factory=list)
    persisted: list[dict] = field(default_factory=list)
    kernel: Kernel = None         # type: ignore

    def __post_init__(self) -> None:
        self.kernel = Kernel(self.policy)

    # -- steps 1–4: scope -> retrieve -> extract -> verify (read-only) ------
    def extract_phase(self, document_ids: list[str] | None = None,
                      ts: str = "") -> list[Item]:
        if self.kernel.killed:
            self.status = "killed"
            return []
        # Kernel governs even the read (whitelisted, budgeted)
        p = ActionProposal("read_documents", {"document_ids": document_ids or []})
        d = self.kernel.validate(p)
        assert d.approved
        self.kernel.execute(p, d, lambda: None)

        rows = self.store.retrieve_chunks(self.session, self.matter_id, ts=ts)  # isolation
        if document_ids:
            rows = [r for r in rows if r.data.get("document_id") in document_ids]

        documents = {r.id: r.data["text"] for r in rows}
        verifier = Verifier(documents, self.nli)

        items: list[Item] = []
        for r in rows:
            loc = Locator(r.id, "pdf", char_range=(0, len(r.data["text"])), page=1) \
                if not isinstance(r.data.get("locator"), Locator) else r.data["locator"]
            # verification span is the chunk itself (chunk-id keyed)
            vloc = Locator(r.id, "pdf", char_range=(0, len(r.data["text"])), page=1)
            for item in self.extractor.extract(r.data["text"], loc):
                res = verifier.verify_pair(item.text, vloc)      # doc 13 gate
                if res.passed:
                    item.verified = True
                    items.append(item)
                # failed items are dropped — never proposed (AC3)
        self.items = items
        self.status = "extracted"
        if self.store.audit is not None:
            self.store.audit.append(actor=f"agent:{self.run_id}", actor_kind="agent",
                                    type="extract_obligations", target=self.matter_id,
                                    payload=f"{len(items)} items proposed", ts=ts)
        return items

    # -- steps 5–6: HITL decision -> governed persist ------------------------
    def decide(self, index: int, decision: str, edited_text: str | None = None,
               *, idempotency_key: str | None = None, ts: str = "") -> Item:
        item = self.items[index]
        if decision == "reject":
            item.status = "rejected"                      # nothing executes
            return item
        if decision == "edit" and edited_text is not None:
            item.text = edited_text
            item.status = "edited"
        # approve (or edit) -> persist through the Kernel with HITL approval
        p = ActionProposal("write_extraction_record",
                           {"item": item.text, "type": item.item_type},
                           idempotency_key=idempotency_key)
        d: Decision = self.kernel.validate(p)
        if not d.approved:
            raise KernelReject(d.code, d.reason)
        record = {"item_type": item.item_type, "text": item.text,
                  "party": item.party, "counterparty": item.counterparty,
                  "trigger_or_due": item.trigger_or_due,
                  "citation": {"document_id": item.locator.document_id if item.locator else None,
                               "verified": item.verified},
                  "status": "persisted"}
        self.kernel.execute(p, d, lambda: self.persisted.append(record),
                            approved=True)               # the human decision IS the checkpoint
        item.status = "persisted"
        if self.store.audit is not None:
            self.store.audit.append(actor=self.session.user_id, actor_kind="user",
                                    type="approve_extraction", target=self.run_id,
                                    payload=item.text, ts=ts)
        return item

    def kill(self) -> None:
        self.kernel.kill_run()
        self.status = "killed"
