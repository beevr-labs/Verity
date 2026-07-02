"""Real OCR tests — FR-KM-04, R13. EasyOCR reads back text rendered with PIL
and returns normalized word-region bboxes. Skips when easyocr is absent."""
import io

import pytest

pytest.importorskip("easyocr")

from PIL import Image, ImageDraw, ImageFont


@pytest.fixture(scope="module")
def ocr():
    from beevr.ingest import EasyOcrProvider
    try:
        return EasyOcrProvider(["en"])
    except Exception as ex:
        pytest.skip(f"EasyOCR unavailable: {ex}")


def _render(text: str, size=(900, 160)) -> bytes:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    d.text((30, 50), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_real_ocr_reads_rendered_text_with_bboxes(ocr):
    words = ocr.ocr_page(_render("Borrower shall maintain leverage ratio"))
    joined = " ".join(t for t, _ in words).lower()
    assert "borrower" in joined and "leverage" in joined
    # bboxes normalized to [0,1] — the R13 locator contract
    for _, (x0, y0, x1, y1) in words:
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_real_ocr_scanned_chunk_via_ingest(ocr):
    """Scanned-page path end-to-end: OCR text becomes a pdf_scanned chunk with
    bbox[], searchable like any other chunk (TC-104)."""
    from beevr.audit import AuditLog
    from beevr.ingest import Chunker, ParsedDoc, Unit
    from beevr.store import Store

    page = _render("The Guarantor waives all defenses")
    words = ocr.ocr_page(page)
    unit = Unit(" ".join(t for t, _ in words), 1, scanned=True, words=words)
    chunks = Chunker().chunk(ParsedDoc("pdf", [unit]), "scan-doc")
    assert len(chunks) == 1
    loc = chunks[0].locator
    assert loc.kind == "pdf_scanned" and len(loc.bbox) > 0
    assert "guarantor" in chunks[0].text.lower()

    store = Store(audit=AuditLog())
    store.put_matter("A", client="X", name="MA")
    store.put_chunk("s0", "A", text=chunks[0].text, locator=loc)
    from beevr.store import Session
    hits = store.retrieve_chunks(Session("alice", matter_grants=frozenset({"A"})),
                                 "A", contains="Guarantor")
    assert [h.id for h in hits] == ["s0"]
