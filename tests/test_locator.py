"""Locator model tests — SRS §3.1.5, R13 (bbox for scanned)."""
import pytest

from beevr.locator import Locator, LocatorError

DOCS = {"doc-1": "The Borrower shall maintain a leverage ratio below 3.0x."}


def test_text_pdf_locator_resolves_to_real_span():
    loc = Locator(document_id="doc-1", kind="pdf", page=12, char_range=(4, 12))
    assert loc.resolves_in(DOCS)
    assert loc.snippet(DOCS) == "Borrower"


def test_fabricated_document_does_not_resolve():
    loc = Locator(document_id="ghost", kind="pdf", page=1, char_range=(0, 5))
    assert loc.resolves_in(DOCS) is False


def test_span_out_of_range_does_not_resolve():
    loc = Locator(document_id="doc-1", kind="pdf", page=1, char_range=(0, 9999))
    assert loc.resolves_in(DOCS) is False


def test_scanned_locator_requires_bbox_R13():
    with pytest.raises(LocatorError, match="bbox"):
        Locator(document_id="doc-1", kind="pdf_scanned", page=1, char_range=(0, 5))
    # with bbox it is valid
    loc = Locator(
        document_id="doc-1", kind="pdf_scanned", page=1, char_range=(0, 3),
        bbox=((0.1, 0.2, 0.5, 0.24),),
    )
    assert loc.resolves_in(DOCS)


def test_bbox_must_be_normalized():
    with pytest.raises(LocatorError, match="normalized"):
        Locator(document_id="doc-1", kind="pdf_scanned", page=1, char_range=(0, 3),
                bbox=((0.1, 0.2, 1.5, 0.24),))


def test_email_and_docx_required_fields():
    with pytest.raises(LocatorError):
        Locator(document_id="d", kind="email", char_range=(0, 1))  # missing message_id/part
    with pytest.raises(LocatorError):
        Locator(document_id="d", kind="docx", char_range=(0, 1))  # missing paragraph_index


def test_bad_char_range_rejected():
    with pytest.raises(LocatorError):
        Locator(document_id="d", kind="plaintext", char_range=(5, 2), line_range=(1, 1))
