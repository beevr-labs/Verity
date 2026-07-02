"""Redaction for hybrid escalation — doc 23, FR-ML-04 (leak-rate gate).

The ONLY sanctioned egress is a redacted slice. Bias = OVER-redact when unsure
(a missed span = privilege breach). Multi-layer, union of detectors:
  - deterministic: email, phone, IBAN (mod-97), card (Luhn), long account nums
  - matter-aware dictionary: the known party names of the matter
  - adversarial normalization first: strip zero-width, fold homoglyphs, join
    digits split by spaces/hyphens/newlines, tolerate spaced-out party letters

`leak_rate` is the release-gate metric (TC-510): sensitive values still present
in the redacted output / total sensitive values. Target = 0 on the adversarial
class before hybrid escalation may be enabled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿­"), None)
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "х": "x", "ѕ": "s", "і": "i",
})


def _fold(text: str) -> str:
    return text.translate(_ZERO_WIDTH).translate(_HOMOGLYPHS)


def _join_digits(text: str) -> str:
    # "4111 1111", "4111-\n1111" -> "41111111"
    return re.sub(r"(?<=\d)[\s\-]+(?=\d)", "", text)


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_ok(candidate: str) -> bool:
    s = candidate.upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearranged)
    return int(num) % 97 == 1


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+\d[\d\s\-]{6,}\d")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_DIGITS = re.compile(r"\d{8,}")


@dataclass
class RedactionResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    counts: dict[str, int] = field(default_factory=dict)


def _party_pattern(name: str) -> re.Pattern:
    # tolerate whitespace between characters ("A c m e") and case.
    chars = [re.escape(c) for c in name if not c.isspace()]
    return re.compile(r"\s*".join(chars), re.IGNORECASE)


def redact(raw: str, party_names: list[str] | None = None) -> RedactionResult:
    text = _fold(raw)
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    seq = {"PARTY": 0}

    def _sub(pattern: re.Pattern, label: str, s: str, *, dynamic=False) -> str:
        def repl(m: re.Match) -> str:
            counts[label] = counts.get(label, 0) + 1
            if dynamic:
                seq["PARTY"] += 1
                ph = f"[{label}_{seq['PARTY']}]"
            else:
                ph = f"[{label}]"
            mapping[ph] = m.group(0)
            return ph
        return pattern.sub(repl, s)

    # 1. matter parties (before digit-join so names stay intact)
    for name in (party_names or []):
        text = _sub(_party_pattern(name), "PARTY", text, dynamic=True)
    # 2. emails, phones (pre-join so separators still present)
    text = _sub(_EMAIL, "EMAIL", text)
    text = _sub(_PHONE, "PHONE", text)
    # 3. join obfuscated digit runs, then structured numbers
    text = _join_digits(text)
    text = _sub(_IBAN, "IBAN", text)

    def _num_repl(m: re.Match) -> str:
        v = m.group(0)
        label = "CARD" if luhn_ok(v) else "ACCOUNT"
        counts[label] = counts.get(label, 0) + 1
        ph = f"[{label}]"
        mapping[ph] = v
        return ph
    text = _DIGITS.sub(_num_repl, text)

    return RedactionResult(text=text, mapping=mapping, counts=counts)


def _norm_value(v: str) -> str:
    return _join_digits(_fold(v)).replace(" ", "").lower()


def leak_rate(redacted_text: str, sensitive_values: list[str]) -> float:
    """Fraction of sensitive values still recoverable from the redacted output."""
    if not sensitive_values:
        return 0.0
    hay = _join_digits(_fold(redacted_text)).replace(" ", "").lower()
    leaked = sum(1 for v in sensitive_values if _norm_value(v) and _norm_value(v) in hay)
    return leaked / len(sensitive_values)
