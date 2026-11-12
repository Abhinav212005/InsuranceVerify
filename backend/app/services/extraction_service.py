"""Runs field extraction for a document and stores one row per expected field."""
from sqlalchemy.orm import object_session

from app.core.errors import AppError
from app.db.models import Document, ExtractedInformation
from app.extraction import extractors
from app.extraction.schema import specs


def _pages(doc: Document) -> list[dict]:
    return [
        {"page": p.page_number, "text": p.ocr_text or "", "words": p.ocr_words_json or [], "avg_confidence": p.avg_confidence}
        for p in doc.pages
    ]


def extract_document(doc: Document, category: str | None = None) -> list[ExtractedInformation]:
    """(Re)extracts fields for `category` (default: the document's category).

    Values a person has edited are never overwritten: they are kept if the field exists in the
    category's schema. Everything else is replaced. Fields that were not found are stored with
    an empty value, so the review screen can list everything that is expected.
    """
    category = category or doc.doc_category
    pages = _pages(doc)
    if not any(p["text"].strip() for p in pages):
        raise AppError("NO_TEXT", "This document has no OCR text yet. Run OCR first.", 409)

    wanted = {s.name for s in specs(category)}
    keep = [f for f in doc.fields if f.is_edited and f.field_name in wanted]
    session = object_session(doc)
    for f in list(doc.fields):
        if f not in keep:
            doc.fields.remove(f)
    if session is not None:
        session.flush()  # deletes must hit the database before inserts (unique document+field)

    kept_names = {f.field_name for f in keep}
    for fld in extractors.extract(category, pages):
        if fld.name in kept_names:
            continue
        doc.fields.append(
            ExtractedInformation(
                field_name=fld.name,
                field_value=fld.value,
                original_value=fld.value,
                confidence=fld.confidence,
                source_page=fld.page,
                # column predates the richer metadata: it holds the box plus how/where the value was found
                bbox_json={"bbox": fld.bbox, "method": fld.method, "source": fld.source, "note": fld.note},
                is_edited=False,
            )
        )
    order = {s.name: i for i, s in enumerate(specs(category))}
    doc.fields.sort(key=lambda f: order.get(f.field_name, 99))
    return doc.fields
