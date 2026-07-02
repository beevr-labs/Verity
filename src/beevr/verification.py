"""Citation verification — doc 13, FR-SF-01/02 (the anti-hallucination gate).

Two stages, BOTH must pass, for every claim->citation pair:
  A. Locator resolution (grounding): the citation must resolve to a real span
     in an in-matter immutable original. Fabricated docs/pages die here.
  B. Entailment: an NLI model must say the span entails the claim, AND a lexical
     guard must hold: pass iff entail >= tau_e AND lexical_overlap >= tau_l.

Answer-assembly rule: a claim survives only if >=1 citation passes A+B. If no
claim survives and the question needed evidence -> ABSTAIN ("insufficient
evidence"). A citation is NEVER shown unless verified => fabricated rate = 0.

The NLI model is injected (a callable premise, hypothesis -> score in [0,1]) so
this logic is testable offline; production wires a local cross-encoder (doc 14).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .locator import Locator

NLI = Callable[[str, str], float]  # (premise_span, hypothesis_claim) -> [0,1]

_WORD = re.compile(r"\w+", re.UNICODE)


def lexical_overlap(claim: str, span: str) -> float:
    """Fraction of claim tokens present in the span (0..1)."""
    c = {w.lower() for w in _WORD.findall(claim)}
    if not c:
        return 0.0
    s = {w.lower() for w in _WORD.findall(span)}
    return len(c & s) / len(c)


@dataclass(frozen=True)
class PairResult:
    passed: bool
    stage_a: bool          # grounded (locator resolved)
    entailment: float
    lexical: float


@dataclass
class Claim:
    text: str
    citations: list[Locator] = field(default_factory=list)


@dataclass
class ShownCitation:
    locator: Locator
    snippet: str
    verified: bool = True   # only verified citations are ever shown


@dataclass
class ShownClaim:
    text: str
    citations: list[ShownCitation]


@dataclass
class AnswerResult:
    claims: list[ShownClaim]
    abstained: bool
    confidence: str         # "high" | "medium" | "low" | "insufficient"


# Clause segmentation for NLI premises. Two measured constraints (EDGAR dev set):
#  * a naive comma split cuts entity names ("BANK OF AMERICA, N.A.") and dates
#    ("June 5, 2012") -> over-abstention (G1/G4 regressions);
#  * merging adjacent segments re-mixes two entities into one premise -> the
#    entity-confusion trap passes again (D1 regression).
# So: protect entity-suffix and date commas, then split ONLY at entity
# boundaries — a comma followed by an ALL-CAPS word (how agreements introduce
# the next party) — or a semicolon. No pair merging.
_PROTECT = re.compile(
    r",(?=\s*(?:INC\b\.?|LLC\b|LTD\b\.?|CORP\b\.?|CO\b\.?|N\.A\b\.?|"
    r"L\.L\.C\b\.?|L\.P\b\.?|S\.A\b\.?|\d{4}\b))", re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(
    r"(?:(?<=,)\s+(?=[A-Z]{2,}))"        # comma before an ALL-CAPS party name
    r"|(?:(?<=,)\s+and\s+(?=[A-Z]{2,}))" # ", and <PARTY>" list-final party
    r"|(?:(?<=;)\s+)"                    # semicolons
    r"|(?:(?<=\bamong)\s+)"              # "by and among <PARTY>..." (first party
    r"|(?:(?<=\bbetween)\s+)")           #  has no leading comma)

# Template/form text is not evidence: an unexecuted exhibit with fill-in blanks
# ("[__]", "____") cannot ground a factual claim, and NLI scores nonsense-high
# on such text (measured 0.994 on a blank joinder form).
_FORM_BLANK = re.compile(r"\[_+\]|_{3,}")

# Role guard: party roles are DEFINED terms — "KEYBANK ... (\"Lender\")". A
# claim binding an entity to a role must match the definitional parenthetical
# that follows the entity in the premise. NLI alone cannot be trusted here
# (measured: premise '... (\"Lender\")' entailed 'is the Borrower' at 0.985).
_ROLES = ("borrower", "lender", "guarantor", "administrative agent", "agent",
          "landlord", "tenant", "licensor", "licensee", "buyer", "seller",
          "indemnitor", "indemnitee", "supplier", "customer", "trustee")
_ROLE_IN_CLAIM = re.compile(
    r"\b(?:is|as)\s+(?:the\s+|an?\s+)?(" + "|".join(_ROLES) + r")\b", re.IGNORECASE)
_DEF_PAREN = re.compile(r"\(\s*[\"“‘']?\s*(?:the\s+)?([A-Za-z ]{3,40}?)\s*[\"”’']?\s*\)")


def _role_binding_ok(claim: str, premise: str) -> bool:
    """If the claim asserts <Entity> is/as <Role>, the first definitional
    parenthetical after the entity in the premise must not name a DIFFERENT
    role. No entity/role/parenthetical found -> no veto (NLI decides)."""
    m = _ROLE_IN_CLAIM.search(claim)
    if not m:
        return True
    claimed = m.group(1).lower()
    ents = claim_entities(claim)
    if not ents:
        return True
    prem_norm = _norm_entity(premise)
    for ent in ents:
        pos = prem_norm.find(ent)
        if pos < 0:
            continue
        # map back roughly: scan definitional parentheticals in the premise and
        # take the first whose normalized position follows the entity
        for dm in _DEF_PAREN.finditer(premise):
            tag_pos = _norm_entity(premise[:dm.start()])
            if len(tag_pos) >= pos + len(ent) - 5:          # tag comes after entity
                tag = dm.group(1).strip().lower()
                if tag in _ROLES and tag != claimed:
                    return False                             # bound to another role
                if tag in _ROLES:
                    return True                              # bound to claimed role
    return True
_SENTINEL = "\x00"


# Entity guard (doc 13 §5 "lexical guard", strengthened): when the claim names
# a legal entity WITH a corporate suffix ("Cenveo, Inc.", "Bank of America,
# N.A."), the supporting premise must contain that exact entity (normalized).
# Rationale: entity identity in agreements is exact-string, and NLI margins on
# near-identical names are too thin to trust (measured: "CENVEO CORPORATION, a
# Delaware corporation" entails "Cenveo, Inc. is a Delaware corporation" at
# 0.978 — threshold-fragile; the entity guard rejects it deterministically).
_ENTITY = re.compile(
    r"\b([A-Z][\w&.-]*(?:[ ,]+[A-Z][\w&.-]*)*?[, ]+"
    r"(?:Inc|Incorporated|LLC|L\.L\.C|Ltd|Limited|Corp|Corporation|"
    r"N\.A|L\.P|S\.A|PLC|plc|Plc|Company|Co|Association|Bank)\.?)(?=[\s,;.)]|$)")


def _norm_entity(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def claim_entities(claim: str) -> list[str]:
    return [_norm_entity(m.group(1)) for m in _ENTITY.finditer(claim)]


def _entities_present(claim: str, premise: str) -> bool:
    prem = _norm_entity(premise)
    return all(e in prem for e in claim_entities(claim))


def _clause_premises(span: str, min_words: int = 3) -> list[str]:
    """Candidate premises at clause granularity (doc 13 §5). The WHOLE span is
    deliberately NOT a candidate: long appositive lists falsely entail
    entity-confusion claims (measured: D1/whole=0.986 vs best-clause=0.905)."""
    protected = _PROTECT.sub(_SENTINEL, span)
    segs = [s.replace(_SENTINEL, ",").strip()
            for s in _CLAUSE_SPLIT.split(protected)]
    segs = [s for s in segs if len(s.split()) >= min_words]
    return segs if len(segs) > 1 else [span]


class Verifier:
    def __init__(self, documents: dict[str, str], nli: NLI,
                 tau_e: float = 0.85, tau_l: float = 0.2,
                 clause_premises: bool = False):
        self.documents = documents
        self.nli = nli
        self.tau_e = tau_e
        self.tau_l = tau_l
        self.clause_premises = clause_premises

    def verify_pair(self, claim_text: str, locator: Locator) -> PairResult:
        if not locator.resolves_in(self.documents):          # Stage A
            return PairResult(False, False, 0.0, 0.0)
        span = locator.snippet(self.documents)
        premises = _clause_premises(span) if self.clause_premises else [span]
        best_ent, best_lex, passed = 0.0, 0.0, False
        for prem in premises:
            lex = lexical_overlap(claim_text, prem)          # Stage B guard
            if lex < self.tau_l:
                continue                                     # cheap pre-filter
            if _FORM_BLANK.search(prem):                     # template veto
                continue                                     # blanks aren't evidence
            if not _entities_present(claim_text, prem):      # entity guard
                continue                                     # wrong/absent entity
            if not _role_binding_ok(claim_text, prem):       # role guard
                continue                                     # bound to another role
            ent = self.nli(prem, claim_text)                 # Stage B entailment
            if ent > best_ent:
                best_ent, best_lex = ent, lex
            if ent >= self.tau_e and lex >= self.tau_l:
                passed = True
                break                                        # first passing clause wins
        return PairResult(passed, True, best_ent, best_lex)

    def assemble(self, claims: list[Claim], *, needs_evidence: bool = True) -> AnswerResult:
        shown: list[ShownClaim] = []
        for claim in claims:
            good: list[ShownCitation] = []
            for loc in claim.citations:
                r = self.verify_pair(claim.text, loc)
                if r.passed:
                    good.append(ShownCitation(loc, loc.snippet(self.documents)))
            if good:                       # claim survives only with >=1 verified cite
                shown.append(ShownClaim(claim.text, good))

        if not shown and needs_evidence:
            return AnswerResult([], abstained=True, confidence="insufficient")
        conf = "high" if len(shown) == len(claims) and shown else "medium" if shown else "low"
        return AnswerResult(shown, abstained=False, confidence=conf)
