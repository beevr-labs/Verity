"""Matter briefing — proactive "what does the lawyer need to do" per matter.

Turns Verity from reactive (lawyer asks) to proactive (system surfaces):
  1. DEADLINE RADAR — every date/trigger found in the matter's documents,
     sorted by urgency, each with a citation back to its source chunk.
     overdue / due-soon (<=90d) / upcoming buckets.
  2. ACTION ITEMS — operational nudges derived from system state: extraction
     proposals awaiting HITL review, failed ingestions, no extraction run yet.
  3. SUGGESTED QUESTIONS — content-aware prompts for the Q&A box. Rule-based
     on detected signals (covenants, termination, guaranty type, governing law,
     parties...). Safe by construction: a clicked suggestion runs through the
     normal verified Q&A — a bad suggestion can only abstain, never fabricate.

UPL discipline (FR-SF-03): the briefing points AT the documents ("found /
not found / due on"), it never advises ("you should...").

`today` is injected (deterministic tests); production passes date.today().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from .store import Session, Store

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

_DATE_RX = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+(\d{4})\b"
    r"|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b", re.IGNORECASE)

_DEADLINE_CUE = re.compile(
    r"\b(terminat\w+|expir\w+|renew\w+|due|deadline|no later than|on or before"
    r"|matur\w+|deliver\w+|notice|within)\b", re.IGNORECASE)


def _parse_dates(text: str) -> list[tuple[date, str]]:
    out = []
    for m in _DATE_RX.finditer(text):
        try:
            if m.group(1):      # "5 June 2012"
                d = date(int(m.group(3)), _MONTHS[m.group(2).lower()[:3]], int(m.group(1)))
            elif m.group(4):    # "June 5, 2012"
                d = date(int(m.group(6)), _MONTHS[m.group(4).lower()[:3]], int(m.group(5)))
            else:               # ISO
                d = date(int(m.group(7)), int(m.group(8)), int(m.group(9)))
            out.append((d, m.group(0)))
        except ValueError:
            continue
    return out


@dataclass
class Deadline:
    due: date
    days_left: int
    status: str                 # overdue | due_soon | upcoming
    context: str                # the sentence it came from
    chunk_id: str               # citation — click-through to source
    date_text: str


@dataclass
class Briefing:
    deadlines: list[Deadline] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)


# --- suggested questions: content signal -> lawyer question ------------------
_SIGNALS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bleverage ratio|financial covenant|shall maintain\b", re.I),
     "What financial covenants must the Borrower maintain?"),
    (re.compile(r"\bterminat\w+|expir\w+", re.I),
     "When does this agreement terminate, and what are the renewal terms?"),
    (re.compile(r"\bguaranty of payment|guaranty of collection|Guarantor\b", re.I),
     "Is this a guaranty of payment or of collection, and what defenses are waived?"),
    (re.compile(r"\bgoverned by|governing law\b", re.I),
     "What is the governing law and jurisdiction?"),
    (re.compile(r"\bevents? of default|cross-default\b", re.I),
     "What are the events of default, and is there a cross-default provision?"),
    (re.compile(r"\bassign\w+|successors\b", re.I),
     "Can this agreement be assigned, and is consent required?"),
    (re.compile(r"\bconfidential\w*\b", re.I),
     "What are the confidentiality obligations and their carve-outs?"),
    (re.compile(r"\bindemnif\w+\b", re.I),
     "Who indemnifies whom, and what is the scope of indemnification?"),
    (re.compile(r"\bsanction\w*|OFAC|anti-money|AML|FCPA\b", re.I),
     "What sanctions/AML representations does the agreement contain?"),
    (re.compile(r"\bnotice\b", re.I),
     "What notice requirements apply, and to whom must notice be given?"),
]

_GENERIC = [
    "Who are the parties and what are their roles?",
    "What are the key obligations of each party?",
]


def build_briefing(store: Store, session: Session, matter_id: str, *,
                   today: date, agent_runs: dict | None = None,
                   ts: str = "") -> Briefing:
    rows = store.retrieve_chunks(session, matter_id, ts=ts)   # RBAC + isolation
    b = Briefing()

    # 1. deadline radar — dates near a deadline cue word, cited to their chunk
    seen: set[tuple[date, str]] = set()
    for r in rows:
        text = r.data["text"]
        if not _DEADLINE_CUE.search(text):
            continue
        for d, raw in _parse_dates(text):
            key = (d, r.id)
            if key in seen:
                continue
            seen.add(key)
            days = (d - today).days
            status = "overdue" if days < 0 else "due_soon" if days <= 90 else "upcoming"
            b.deadlines.append(Deadline(d, days, status, text[:160], r.id, raw))
    b.deadlines.sort(key=lambda x: x.due)

    # 2. action items from system state
    docs = store.list_documents(session, matter_id, ts=ts)
    failed = [d.id for d in docs if d.data.get("ingest_status") == "failed"]
    if failed:
        b.actions.append(f"{len(failed)} document(s) failed ingestion — review and re-upload")
    matter_runs = [run for run in (agent_runs or {}).values()
                   if getattr(run, "matter_id", None) == matter_id]
    pending = sum(1 for run in matter_runs for it in run.items
                  if it.status == "proposed")
    if pending:
        b.actions.append(f"{pending} extracted item(s) awaiting your HITL review")
    if rows and not matter_runs:
        b.actions.append("Obligation extraction has not been run on this matter yet")
    overdue = [d for d in b.deadlines if d.status == "overdue"]
    soon = [d for d in b.deadlines if d.status == "due_soon"]
    if overdue:
        b.actions.append(f"{len(overdue)} dated item(s) in the documents are in the past — verify status")
    if soon:
        b.actions.append(f"{len(soon)} dated item(s) fall within 90 days — check required action")

    # 3. suggested questions from content signals (order = signal frequency)
    joined = " \n".join(r.data["text"] for r in rows)
    scored = []
    for rx, q in _SIGNALS:
        n = len(rx.findall(joined))
        if n:
            scored.append((n, q))
    b.suggested_questions = [q for _, q in sorted(scored, key=lambda x: -x[0])][:6]
    if rows:
        b.suggested_questions += [q for q in _GENERIC
                                  if q not in b.suggested_questions][:8 - len(b.suggested_questions)]
    return b
