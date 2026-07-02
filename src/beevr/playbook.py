"""Contract review vs playbook — US-303b, FR-AG-08/09 (TC-306/307).

A playbook is the organization's standard negotiating positions. The review
checks a contract clause-by-clause against each rule and flags:

  * COMPLIES  — a clause matches the standard position (cited)
  * DEVIATES  — a clause contradicts it (cited, with rationale + a suggested
                redline that is a PROPOSAL — never auto-applied, FR-AG-09)
  * ABSENT    — no relevant clause found ("not found in the reviewed
                documents" — a pointer, not a legal conclusion)

Grounding discipline (TC-306): a flag is only shown if its `quote` is an
EXACT substring of the cited chunk — an LLM-invented quote dies here, same
spirit as citation verification Stage A.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .fusion import fuse_ids
from .store import Session, Store

JUDGE_PROMPT = """You are reviewing a contract clause against a company playbook rule.

PLAYBOOK RULE: {position}

CLAUSE:
{clause}

Output ONLY JSON:
{{"relevant": true/false, "complies": true/false,
  "quote": "<exact words copied verbatim from the CLAUSE that decide this>",
  "rationale": "<one sentence: how the quote complies with or deviates from the rule>"}}
- "relevant": false if the clause has nothing to do with the rule's topic.
- "quote" MUST be copied character-for-character from the CLAUSE. Never paraphrase it.
- Restate what the clause says; no legal advice.

JSON:"""

REDLINE_PROMPT = """A contract clause deviates from the company playbook.

PLAYBOOK RULE: {position}
CLAUSE QUOTE: {quote}

Draft ONE replacement sentence bringing the clause in line with the rule.
Output ONLY the replacement sentence, no commentary."""

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class Rule:
    id: str
    topic: str
    standard_position: str      # the assertion the contract should satisfy
    severity: str = "medium"    # low | medium | high
    keywords: str = ""          # retrieval hint; defaults to the position text


# A starter playbook for the FI beachhead — deployments replace this via
# /admin/playbooks (doc 11 §3.6). Positions are assertions, checkable per clause.
FI_DEFAULT_PLAYBOOK = [
    Rule("GL-1", "governing law",
         "The agreement must be governed by the laws of the State of New York",
         "high", "governed by construed in accordance with laws"),
    Rule("GT-1", "guaranty type",
         "Any guaranty must be a guaranty of payment, not merely of collection",
         "high", "guaranty of payment collection"),
    Rule("CF-1", "confidentiality return",
         "Confidential information must be returned or destroyed upon written request",
         "medium", "confidential information return destroy written request"),
    Rule("AS-1", "assignment consent",
         "Assignment requires the prior written consent of the other party",
         "medium", "assign assignment consent successors"),
    Rule("NT-1", "notice",
         "Notices must be given in writing",
         "low", "notice notices given writing"),
]


@dataclass
class Flag:
    rule_id: str
    topic: str
    severity: str
    status: str                 # complies | deviates | absent
    quote: str = ""
    rationale: str = ""
    chunk_id: str = ""          # citation (TC-306)
    suggested_redline: str = "" # PROPOSAL only — never auto-applied (TC-307)


def _judge(llm, position: str, clause: str) -> dict | None:
    raw = llm.generate(JUDGE_PROMPT.format(position=position, clause=clause),
                       max_new_tokens=300)
    m = _JSON_OBJ.search(raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def review(store: Store, session: Session, matter_id: str, *, llm,
           rules: list[Rule] | None = None, embedder=None, top_n: int = 4,
           ts: str = "") -> list[Flag]:
    rules = rules if rules is not None else FI_DEFAULT_PLAYBOOK
    rows = store.retrieve_chunks(session, matter_id, ts=ts)   # RBAC + isolation
    by_id = {r.id: r.data["text"] for r in rows}
    flags: list[Flag] = []

    for rule in rules:
        probe = rule.keywords or rule.standard_position
        p_tokens = {t.lower() for t in probe.split()}

        def overlap(r):
            return len(p_tokens & {t.lower().strip(".,;()") for t in r.data["text"].split()})

        if embedder is not None and rows and rows[0].data.get("embedding"):
            qv = embedder.embed(rule.standard_position)
            def cos(r):
                v = r.data["embedding"]
                num = sum(a * b for a, b in zip(qv, v))
                den = (sum(a * a for a in qv) ** 0.5) * (sum(b * b for b in v) ** 0.5)
                return num / den if den else 0.0
            vec = [r.id for r in sorted(rows, key=lambda r: -cos(r))]
        else:
            vec = [r.id for r in sorted(rows, key=lambda r: -overlap(r))]
        lex = [r.id for r in sorted(rows, key=lambda r: -overlap(r)) if overlap(r) > 0]
        top = fuse_ids([vec, lex], top_n=top_n)

        found: Flag | None = None
        for cid in top:
            clause = by_id[cid]
            d = _judge(llm, rule.standard_position, clause)
            if not d or not d.get("relevant"):
                continue
            quote = str(d.get("quote", "")).strip()
            # TC-306 grounding: the quote must exist verbatim in the clause
            if not quote or quote not in clause:
                continue                          # invented quote -> discarded
            status = "complies" if d.get("complies") else "deviates"
            flag = Flag(rule.id, rule.topic, rule.severity, status,
                        quote=quote, rationale=str(d.get("rationale", ""))[:300],
                        chunk_id=cid)
            if status == "deviates":
                # suggested redline is a PROPOSAL (FR-AG-09/TC-307): returned
                # for HITL review; nothing is ever applied to any document.
                flag.suggested_redline = llm.generate(
                    REDLINE_PROMPT.format(position=rule.standard_position,
                                          quote=quote),
                    max_new_tokens=120).strip()
                found = flag
                break                             # a deviation decides the rule
            found = found or flag                 # keep first compliant hit
        flags.append(found or Flag(rule.id, rule.topic, rule.severity, "absent",
                                   rationale="No relevant clause found in the "
                                             "reviewed documents."))
        if store.audit is not None:
            f = flags[-1]
            store.audit.append(actor=session.user_id, actor_kind="user",
                               type="playbook_check", target=rule.id,
                               payload=f"{f.status}:{f.chunk_id}", ts=ts)
    return flags
