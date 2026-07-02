"""Ingestion — doc 22 §1.2, FR-KM-02/04/05, R13.

Real parsers for the formats we can handle dependency-light:
  * .txt/.csv/.md  -> stdlib
  * .eml           -> stdlib `email`
  * .docx          -> python-docx
  * .pdf           -> pypdf (text layer); pages with no text layer are "scanned"
                      and routed to a pluggable OcrProvider that returns word
                      bounding boxes, so the chunk carries a `pdf_scanned`
                      Locator with bbox[] (R13).

The OCR engine and the embedding model are injected (providers), so this runs
offline with stubs; production wires Tesseract-class OCR and bge-m3 embeddings
(doc 14) without touching the parse/chunk/store flow.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Protocol

from .locator import BBox, Locator
from .store import Store


class UnsupportedType(Exception):
    pass


# --------------------------------------------------------------------------
# Providers (pluggable)
# --------------------------------------------------------------------------
class OcrProvider(Protocol):
    def ocr_page(self, image: bytes) -> list[tuple[str, BBox]]:
        """Return (word, normalized-bbox) pairs for a scanned page image."""


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class StubEmbedder:
    """Deterministic, offline stand-in for bge-m3 (real dim = 1024, doc 14)."""
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[: self.dim]]


class EasyOcrProvider:
    """Real OCR (EasyOCR on torch/CUDA): image bytes -> (text, normalized bbox)
    per detected region. Produces the bbox[] a `pdf_scanned` Locator needs for
    visual highlighting (R13). Weights download once, then run offline."""

    def __init__(self, langs: list[str] | None = None):
        import easyocr
        self.reader = easyocr.Reader(langs or ["en"], verbose=False)

    def ocr_page(self, image: bytes) -> list[tuple[str, BBox]]:
        import io as _io

        from PIL import Image
        img = Image.open(_io.BytesIO(image))
        w, h = img.size
        import numpy as np
        results = self.reader.readtext(np.array(img))
        out: list[tuple[str, BBox]] = []
        for quad, text, _conf in results:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            out.append((text, (max(0.0, min(xs) / w), max(0.0, min(ys) / h),
                               min(1.0, max(xs) / w), min(1.0, max(ys) / h))))
        return out


class StubOcr:
    """Test OCR: lays words on a grid and returns normalized bboxes."""
    def ocr_page(self, image: bytes) -> list[tuple[str, BBox]]:
        words = image.decode("utf-8", "ignore").split()
        out: list[tuple[str, BBox]] = []
        for i, w in enumerate(words):
            y = 0.05 + (i // 10) * 0.03
            x = 0.05 + (i % 10) * 0.09
            out.append((w, (round(x, 3), round(y, 3), round(x + 0.08, 3), round(y + 0.02, 3))))
        return out


# --------------------------------------------------------------------------
# Parsed representation
# --------------------------------------------------------------------------
@dataclass
class Unit:
    """A page (pdf), paragraph (docx), or the whole body (txt/email)."""
    text: str
    index: int                         # page no. (1-based) or paragraph index
    scanned: bool = False
    words: list[tuple[str, BBox]] = field(default_factory=list)  # scanned pages


@dataclass
class ParsedDoc:
    kind: str                          # pdf|docx|email|plaintext
    units: list[Unit]
    message_id: str | None = None      # email


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def _reflow(text: str) -> str:
    """Join hard line-wraps (single \\n inside a paragraph) so sentences broken
    by source formatting stay whole — a wrapped sentence loses its subject and
    defeats NLI verification (measured: 'governed by Ohio law' chunk scored
    0.131 because its subject sat on the previous wrapped line)."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def _parse_text(data: bytes) -> ParsedDoc:
    return ParsedDoc("plaintext", [Unit(_reflow(data.decode("utf-8", "replace")), 0)])


def _parse_email(data: bytes) -> ParsedDoc:
    import email
    from email import policy
    msg = email.message_from_bytes(data, policy=policy.default)
    body = msg.get_body(preferencelist=("plain",))
    text = body.get_content() if body else (msg.get_content() if not msg.is_multipart() else "")
    header = f"From: {msg['from']}\nTo: {msg['to']}\nSubject: {msg['subject']}\n\n"
    return ParsedDoc("email", [Unit(header + text, 0)],
                     message_id=msg.get("message-id"))


def _parse_docx(data: bytes) -> ParsedDoc:
    import docx
    d = docx.Document(io.BytesIO(data))
    units = [Unit(p.text, i) for i, p in enumerate(d.paragraphs) if p.text.strip()]
    return ParsedDoc("docx", units or [Unit("", 0)])


def _parse_pdf(data: bytes, ocr: OcrProvider | None) -> ParsedDoc:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    units: list[Unit] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            units.append(Unit(text, i + 1))
        elif ocr is not None:                              # scanned page -> OCR (R13)
            words = ocr.ocr_page(f"page-{i+1}".encode())   # real: rasterize page
            units.append(Unit(" ".join(w for w, _ in words), i + 1,
                              scanned=True, words=words))
        else:
            units.append(Unit("", i + 1, scanned=True))
    return ParsedDoc("pdf", units)


_EXT = {"txt": _parse_text, "csv": _parse_text, "md": _parse_text}


def parse(data: bytes, filename: str, *, ocr: OcrProvider | None = None) -> ParsedDoc:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _EXT:
        return _EXT[ext](data)
    if ext == "eml":
        return _parse_email(data)
    if ext == "docx":
        return _parse_docx(data)
    if ext == "pdf":
        return _parse_pdf(data, ocr)
    raise UnsupportedType(f"unsupported file type: .{ext or '(none)'}")


# --------------------------------------------------------------------------
# Chunker (sentence/clause segmentation) -> chunks with real Locators
# --------------------------------------------------------------------------
# Legal-aware sentence split: a "." does NOT end a sentence when it belongs to
#   * an abbreviation common in agreements: No., Art., Sec., Inc., Corp., N.A.,
#     U.S., v. (case cites), single capitals ("N.", "A.")
#   * a section/exhibit number: "Section 2.01", "Sections 4.", "Exhibit 10.1"
#   * a decimal: "3.0x", "1.25"
# (Measured failure: the naive [.;!?] split fragmented "Section 2.01" and
#  "BANK OF AMERICA, N.A." — the root cause of fragment chunks in the real
#  EDGAR run; see PROGRESS "Real-document validation".)
_SENT = re.compile(r"[^.;!?\n]+[.;!?]?")
_ABBREV_TAIL = re.compile(
    r"(?:\b(?:No|Nos|Art|Arts|Sec|Secs|Section|Sections|Exhibit|Schedule|Clause|"
    r"Inc|Corp|Co|Ltd|LLC|L\.L\.C|N\.A|U\.S|v|vs)\.$)"
    r"|(?:\b[A-Z]\.$)"          # single-capital initials: "N.", "A."
    r"|(?:\d\.$)",              # decimal or numbered heading: "2." in "2.01"
    re.IGNORECASE)


@dataclass
class Chunk:
    text: str
    locator: Locator


def _segments(text: str) -> list[tuple[int, int]]:
    """Sentence spans, merging fragments produced by legal abbreviations
    ("Section 2.01", "N.A.", "3.0x") back into one span."""
    raw = [(m.start(), m.end()) for m in _SENT.finditer(text) if m.group().strip()]
    if not raw:
        return [(0, len(text))] if text else []
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged:
            ps, pe = merged[-1]
            prev = text[ps:pe].rstrip()
            gap = text[pe:s]
            # merge when the previous span ends in an abbreviation/decimal dot,
            # or this span starts lowercase/digit hard against the previous one
            nxt = text[s:e].lstrip()
            if (_ABBREV_TAIL.search(prev) or
                    (not gap.strip() and nxt[:1].islower()) or
                    (prev.endswith(".") and nxt[:1].isdigit() and not gap.strip())):
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    return merged


class Chunker:
    def chunk(self, parsed: ParsedDoc, document_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for unit in parsed.units:
            if unit.scanned:
                # page-granular for scanned images; bbox = all word boxes (R13)
                if not unit.text:
                    continue
                loc = Locator(document_id, "pdf_scanned", char_range=(0, len(unit.text)),
                              page=unit.index, bbox=tuple(b for _, b in unit.words))
                chunks.append(Chunk(unit.text, loc))
                continue
            for (s, e) in _segments(unit.text):
                span = unit.text[s:e].strip()
                if not span:
                    continue
                loc = self._locator(parsed, document_id, unit, s, e)
                chunks.append(Chunk(span, loc))
        return chunks

    def _locator(self, parsed, document_id, unit, s, e) -> Locator:
        if parsed.kind == "pdf":
            return Locator(document_id, "pdf", char_range=(s, e), page=unit.index)
        if parsed.kind == "docx":
            return Locator(document_id, "docx", char_range=(s, e),
                           paragraph_index=unit.index)
        if parsed.kind == "email":
            return Locator(document_id, "email", char_range=(s, e),
                           message_id=parsed.message_id or "unknown", part=0)
        line = unit.text.count("\n", 0, s)
        return Locator(document_id, "plaintext", char_range=(s, e),
                       line_range=(line, line))


# --------------------------------------------------------------------------
# Full ingest of one document -> chunks written to the Store
# --------------------------------------------------------------------------
@dataclass
class IngestReport:
    document_id: str
    kind: str
    chunks: int
    status: str                        # done|failed
    error: str | None = None


def ingest_document(store: Store, matter_id: str, document_id: str, data: bytes,
                    filename: str, *, ocr: OcrProvider | None = None,
                    embedder: Embedder | None = None) -> IngestReport:
    embedder = embedder or StubEmbedder()
    ocr = ocr or StubOcr()
    try:
        parsed = parse(data, filename, ocr=ocr)
        chunks = Chunker().chunk(parsed, document_id)
    except UnsupportedType as ex:
        return IngestReport(document_id, "?", 0, "failed", str(ex))

    for i, ch in enumerate(chunks):
        store.put_chunk(f"{document_id}-c{i}", matter_id, text=ch.text,
                        locator=ch.locator, embedding=embedder.embed(ch.text),
                        document_id=document_id)
    return IngestReport(document_id, parsed.kind, len(chunks), "done")
