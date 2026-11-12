"""Classifies a document from its OCR text and records the outcome."""
from datetime import datetime, timezone

from app.classification import classifier
from app.core.config import settings
from app.core.errors import AppError
from app.db.models import Document


def document_text(doc: Document) -> str:
    return "\n\n".join(p.ocr_text for p in doc.pages if p.ocr_text)


def classify_document(doc: Document, method: str | None = None) -> dict:
    text = document_text(doc)
    if not text.strip():
        raise AppError("NO_TEXT", "This document has no OCR text yet. Run OCR first.", 409)
    result = classifier.classify(text, method or settings.classifier_method)
    result["classified_at"] = datetime.now(timezone.utc).isoformat()
    doc.doc_category = result["category"]
    doc.category_confidence = result["confidence"]
    doc.classification_method = result["method"]
    doc.classification_json = result
    return result
