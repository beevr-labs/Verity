"""Ingestion tests — FR-KM-02/04/05, TC-103/104 analogues, R13.

Real parsers: txt/csv (stdlib), .eml (stdlib email), .docx (python-docx),
.pdf (pypdf) — including a hand-built real PDF with a text layer, and a
scanned (no-text-layer) PDF routed through the OCR provider to produce a
`pdf_scanned` Locator with word bboxes (R13).
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient

from beevr.api import AppState, create_app
from beevr.audit import AuditLog
from beevr.ingest import UnsupportedType, ingest_document, parse
from beevr.store import Session, Store


# ---- helpers --------------------------------------------------------------
def _make_pdf(page_texts: list[str | None]) -> bytes:
    """Build a real (minimal) PDF. `None` = a scanned page (no text layer)."""
    objs: list[bytes] = []
    n_pages = len(page_texts)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    font_num = 3 + 2 * n_pages
    for i, text in enumerate(page_texts):
        content_num = 3 + 2 * i + 1
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                     f"/Contents {content_num} 0 R /Resources << /Font "
                     f"<< /F1 {font_num} 0 R >> >> >>").encode())
        if text is None:
            stream = b""                                  # scanned: empty content
        else:
            esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode()
                    + b" >>\nstream\n" + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_at = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
              f"startxref\n{xref_at}\n%%EOF".encode())
    return out.getvalue()


def _make_docx(paragraphs: list[str]) -> bytes:
    import docx
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _make_eml(subject: str, body: str) -> bytes:
    return (f"From: counsel@bank-x.example\nTo: legal@bank-x.example\n"
            f"Subject: {subject}\nMessage-ID: <m1@bank-x>\n"
            f"Content-Type: text/plain\n\n{body}").encode()


def _store():
    s = Store(audit=AuditLog())
    s.put_matter("A", client="X", name="Facility A")
    return s


# ---- parser tests ----------------------------------------------------------
def test_txt_parse_and_chunk_locators():
    store = _store()
    rep = ingest_document(store, "A", "d1",
                          b"First clause applies. Second clause governs fees.\nThird line here.\n\nNew paragraph starts.",
                          "note.txt")
    # short texts MERGE into one retrieval-sized chunk (min_chars, see Chunker);
    # single \n is REFLOWED to a space so wrapped sentences stay whole for NLI
    assert rep.status == "done" and rep.chunks == 1
    chunk = store.chunks["d1-c0"]
    assert chunk.data["locator"].kind == "plaintext"
    for part in ["First clause applies", "Third line here", "New paragraph starts"]:
        assert part in chunk.data["text"]

    # fine-grained segmentation still works when merging is disabled
    from beevr.ingest import Chunker, parse
    segs = Chunker(min_chars=0).chunk(
        parse(b"First clause applies. Second clause governs fees.\nThird one.", "n.txt"), "d")
    assert len(segs) == 3


def test_eml_real_parse():
    store = _store()
    rep = ingest_document(store, "A", "d2",
                          _make_eml("Covenant waiver", "The waiver expires on 31 March 2027."),
                          "mail.eml")
    assert rep.status == "done" and rep.kind == "email"
    texts = [store.chunks[f"d2-c{i}"].data["text"] for i in range(rep.chunks)]
    assert any("waiver expires" in t for t in texts)
    loc = store.chunks["d2-c0"].data["locator"]
    assert loc.kind == "email" and loc.message_id == "<m1@bank-x>"


def test_docx_real_parse():
    store = _store()
    data = _make_docx(["Section 1. The Borrower shall repay the loan.",
                       "Section 2. Interest accrues at SOFR plus margin."])
    rep = ingest_document(store, "A", "d3", data, "agreement.docx")
    assert rep.status == "done" and rep.kind == "docx" and rep.chunks >= 2
    loc = store.chunks["d3-c0"].data["locator"]
    assert loc.kind == "docx" and loc.paragraph_index == 0


def test_pdf_text_layer_real_parse():
    store = _store()
    data = _make_pdf(["The Guarantor waives all defenses under section 9."])
    rep = ingest_document(store, "A", "d4", data, "guaranty.pdf")
    assert rep.status == "done" and rep.kind == "pdf" and rep.chunks >= 1
    texts = [c.data["text"] for c in store.chunks.values()]
    assert any("Guarantor waives" in t for t in texts)
    loc = store.chunks["d4-c0"].data["locator"]
    assert loc.kind == "pdf" and loc.page == 1


def test_scanned_pdf_gets_bbox_locator_R13():
    store = _store()
    data = _make_pdf([None])                               # 1 page, no text layer
    rep = ingest_document(store, "A", "d5", data, "scan.pdf")
    assert rep.status == "done" and rep.chunks == 1
    loc = store.chunks["d5-c0"].data["locator"]
    assert loc.kind == "pdf_scanned" and loc.page == 1
    assert len(loc.bbox) > 0                               # word boxes present
    assert all(0.0 <= v <= 1.0 for b in loc.bbox for v in b)


def test_unsupported_type_fails_cleanly():
    store = _store()
    rep = ingest_document(store, "A", "d6", b"binary", "malware.exe")
    assert rep.status == "failed" and "unsupported" in rep.error
    with pytest.raises(UnsupportedType):
        parse(b"x", "archive.zip")


def test_legal_abbreviations_not_split():
    """Root-cause fix from the real EDGAR run: 'Section 2.01', 'N.A.', '3.0x'
    must stay inside one segment (fragments broke retrieval + NLI premises)."""
    from beevr.ingest import Chunker, parse
    text = ("The conditions set forth in Sections 4.02 and 4.03 have been satisfied. "
            "BANK OF AMERICA, N.A. acts as Administrative Agent under Section 2.01 hereof. "
            "The Borrower shall maintain a leverage ratio below 3.0x at all times.")
    segs = Chunker(min_chars=0).chunk(parse(text.encode(), "a.txt"), "d8")
    texts = [c.text for c in segs]
    assert len(texts) == 3                                  # not shredded into fragments
    assert any("Sections 4.02 and 4.03" in t for t in texts)
    assert any("N.A. acts as Administrative Agent under Section 2.01" in t for t in texts)
    assert any("3.0x at all times" in t for t in texts)
    # no fragment chunks like "01 hereof" or "A."
    assert all(len(t) > 20 for t in texts)


def test_embeddings_written():
    store = _store()
    ingest_document(store, "A", "d7", b"One clause.", "a.txt")
    emb = store.chunks["d7-c0"].data["embedding"]
    assert isinstance(emb, list) and len(emb) == 8         # stub dim


# ---- end-to-end over the API -----------------------------------------------
@pytest.fixture()
def api():
    state = AppState()
    client = TestClient(create_app(state))
    tok = client.post("/auth/token", json={
        "sub": "alice", "roles": ["user"], "matter_grants": ["A"]}).json()["token"]
    hdr = {"Authorization": f"Bearer {tok}"}
    client.post("/matters", json={"id": "A", "client": "X", "name": "Facility A"},
                headers=hdr)
    return client, hdr


def test_upload_real_docx_then_query_returns_cited_answer(api):
    client, hdr = api
    data = _make_docx(["The Borrower shall maintain a leverage ratio below 3.0x."])
    r = client.post("/matters/A/documents", json={
        "filename": "credit-agreement.docx",
        "content_b64": base64.b64encode(data).decode()}, headers=hdr)
    assert r.status_code == 200 and r.json()["status"] == "done"
    assert r.json()["chunks"] >= 1

    q = client.post("/matters/A/queries", json={
        "question": "leverage ratio the Borrower shall maintain"}, headers=hdr)
    body = q.json()
    assert body["abstained"] is False
    assert "leverage ratio" in body["claims"][0]["citations"][0]["snippet"]

    docs = client.get("/matters/A/documents", headers=hdr).json()["documents"]
    assert docs[0]["ingest_status"] == "done"


def test_upload_unsupported_type_is_422(api):
    client, hdr = api
    r = client.post("/matters/A/documents", json={
        "filename": "virus.exe", "content": "MZ..."}, headers=hdr)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNSUPPORTED_TYPE"
