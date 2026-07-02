"""Citation verification tests — FR-SF-01/02.

Release-gating: TC-205 (fabricated-citation rate = 0 on adversarial set).
TC-206 (unverifiable claim dropped/flagged, not shown as fact).
TC-803 (abstain when insufficient evidence).

NLI is stubbed (offline, deterministic): entailment = lexical overlap, so a span
that actually contains the claim's words "entails" it. Production swaps in a
local cross-encoder (doc 14) without changing this logic.
"""
import pytest

from beevr.locator import Locator
from beevr.verification import Claim, Verifier, lexical_overlap

DOCS = {
    "agr-1": "The Borrower shall maintain a leverage ratio below 3.0x at all times.",
    "agr-2": "This Agreement terminates on 31 March 2027 unless renewed in writing.",
}

# Stub NLI: treat lexical overlap as the entailment score (deterministic, offline).
def stub_nli(span: str, claim: str) -> float:
    return lexical_overlap(claim, span)


def _v() -> Verifier:
    return Verifier(DOCS, stub_nli, tau_e=0.85, tau_l=0.2)


def test_grounded_supported_claim_is_shown():
    v = _v()
    loc = Locator("agr-1", "pdf", char_range=(0, 58), page=1)
    claim = Claim("The Borrower shall maintain a leverage ratio below 3.0x",
                  citations=[loc])
    res = v.assemble([claim])
    assert not res.abstained
    assert len(res.claims) == 1 and res.claims[0].citations[0].verified


@pytest.mark.release_gating
def test_fabricated_citation_never_shown_TC205():
    v = _v()
    # (a) cites a non-existent document; (b) real span that does NOT support claim.
    fabricated_doc = Claim(
        "The Borrower may sell the collateral without notice",
        citations=[Locator("ghost-doc", "pdf", char_range=(0, 20), page=1)])
    unsupported_real_span = Claim(
        "The interest rate is fixed at 5 percent",   # not in agr-2's span
        citations=[Locator("agr-2", "pdf", char_range=(0, 58), page=1)])
    res = v.assemble([fabricated_doc, unsupported_real_span])
    # Nothing verifiable -> abstain, and ZERO citations surfaced.
    assert res.abstained is True
    assert res.confidence == "insufficient"
    shown_citations = [c for claim in res.claims for c in claim.citations]
    assert shown_citations == []          # fabricated-citation rate = 0


def test_unverifiable_claim_dropped_but_verifiable_kept_TC206():
    v = _v()
    good = Claim("This Agreement terminates on 31 March 2027",
                 citations=[Locator("agr-2", "pdf", char_range=(0, 62), page=1)])
    bad = Claim("A penalty of 10 percent applies on default",
                citations=[Locator("agr-2", "pdf", char_range=(0, 62), page=1)])
    res = v.assemble([good, bad])
    assert not res.abstained
    shown_texts = [c.text for c in res.claims]
    assert "This Agreement terminates on 31 March 2027" in shown_texts
    assert "A penalty of 10 percent applies on default" not in shown_texts


def test_abstain_when_no_evidence_TC803():
    v = _v()
    res = v.assemble([], needs_evidence=True)
    assert res.abstained and res.confidence == "insufficient"


def test_stage_a_kills_out_of_range_span():
    v = _v()
    r = v.verify_pair("anything", Locator("agr-1", "pdf", char_range=(0, 9999), page=1))
    assert r.stage_a is False and r.passed is False
