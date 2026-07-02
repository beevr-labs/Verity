"""Redaction tests — FR-ML-04, TC-510 (leak rate = 0 on adversarial class).

The escalation feature cannot be enabled unless leak_rate == 0 here.
"""
import pytest

from beevr.redaction import iban_ok, leak_rate, luhn_ok, redact


def test_luhn_and_iban_validators():
    assert luhn_ok("4111111111111111")      # test Visa
    assert not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432")
    assert not iban_ok("GB00WEST12345698765432")


def test_structured_pii_redacted():
    r = redact("Wire to IBAN GB82WEST12345698765432, card 4111 1111 1111 1111, "
               "email john@acme.com, call +1 555 123 4567.")
    assert "GB82WEST12345698765432" not in r.text
    assert "4111" not in r.text
    assert "john@acme.com" not in r.text
    assert "[IBAN]" in r.text and "[CARD]" in r.text and "[EMAIL]" in r.text


@pytest.mark.release_gating
def test_adversarial_leak_rate_is_zero_TC510():
    parties = ["Acme Corporation", "Zenith Holdings"]
    # Adversarial: spaced-out digits, hyphen/newline-split card, zero-width in
    # email, spaced party letters, homoglyph in a party name.
    slice_text = (
        "Counterparty A c m e   C o r p o r a t i o n owes Zеnith Holdings.\n"   # homoglyph 'е'
        "Account 4 1 1 1  1 1 1 1  1 1 1 1  1 1 1 1 due; ref 9 8 7 6 5 4 3 2 1.\n"
        "Card 4111-\n1111-1111-1111. Email jo​hn@acme.com. Tel +1 555 123 4567."
    )
    sensitive = [
        "Acme Corporation", "Zenith Holdings",
        "4111111111111111", "987654321",
        "john@acme.com", "+15551234567",
    ]
    r = redact(slice_text, party_names=parties)
    assert leak_rate(r.text, sensitive) == 0.0     # gate: zero leaks


def test_leak_rate_detects_a_miss():
    # A redactor that fails to catch a value must report a non-zero leak rate.
    unredacted = "secret account 12345678 stays here"
    assert leak_rate(unredacted, ["12345678"]) == 1.0


def test_over_redaction_guardrail_short_numbers_kept():
    # Years / short refs should NOT be redacted (utility guardrail).
    r = redact("The 2027 renewal covers section 12 of the 3.0x covenant.")
    assert "2027" in r.text and "12" in r.text
