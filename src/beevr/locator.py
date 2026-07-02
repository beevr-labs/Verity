"""Locator model — SRS §3.1.5 (normative).

A Locator uniquely addresses a source span for a citation across formats.
Every citation MUST carry a Locator that resolves to a highlightable span.
Scanned/OCR'd pages MUST additionally carry word-level bounding boxes (R13).

Maps to: FR-KM-10 (pinpoint citation), doc 11 §3.3 wire format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# A normalized bounding box: [x0, y0, x1, y1] in [0, 1] relative to the page.
BBox = tuple[float, float, float, float]

_KINDS = {"pdf", "pdf_scanned", "docx", "email", "plaintext"}


class LocatorError(ValueError):
    """Raised when a Locator is structurally invalid."""


@dataclass(frozen=True)
class Locator:
    document_id: str
    kind: str
    char_range: tuple[int, int]
    # kind-specific:
    page: int | None = None            # pdf, pdf_scanned
    paragraph_index: int | None = None  # docx
    message_id: str | None = None      # email
    part: int | None = None            # email
    line_range: tuple[int, int] | None = None  # plaintext
    bbox: tuple[BBox, ...] = field(default_factory=tuple)  # pdf_scanned (required)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise LocatorError(f"unknown locator kind: {self.kind!r}")
        if not self.document_id:
            raise LocatorError("document_id is required")
        s, e = self.char_range
        if s < 0 or e < s:
            raise LocatorError(f"invalid char_range {self.char_range!r}")

        if self.kind in ("pdf", "pdf_scanned"):
            if self.page is None or self.page < 1:
                raise LocatorError("pdf locator requires page >= 1")
        if self.kind == "pdf_scanned" and not self.bbox:
            # R13: a char_range alone cannot highlight a raster page.
            raise LocatorError("pdf_scanned locator requires bbox[]")
        if self.kind == "docx" and self.paragraph_index is None:
            raise LocatorError("docx locator requires paragraph_index")
        if self.kind == "email" and (self.message_id is None or self.part is None):
            raise LocatorError("email locator requires message_id and part")
        if self.kind == "plaintext" and self.line_range is None:
            raise LocatorError("plaintext locator requires line_range")

        for b in self.bbox:
            if len(b) != 4 or not all(0.0 <= v <= 1.0 for v in b):
                raise LocatorError(f"bbox must be 4 normalized floats in [0,1]: {b!r}")

    def resolves_in(self, documents: dict[str, str]) -> bool:
        """Stage A grounding: does this locator point to a real span in a real,
        in-matter document? Returns False for fabricated documents/spans.
        `documents` maps document_id -> full text (the immutable original)."""
        text = documents.get(self.document_id)
        if text is None:
            return False  # fabricated document
        s, e = self.char_range
        return e <= len(text) and s < e  # span must fit and be non-empty

    def snippet(self, documents: dict[str, str]) -> str:
        if not self.resolves_in(documents):
            raise LocatorError("locator does not resolve; cannot extract snippet")
        s, e = self.char_range
        return documents[self.document_id][s:e]
