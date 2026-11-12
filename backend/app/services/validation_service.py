"""Gathers the facts about a document (fields, OCR/classification quality, duplicates, matching policy),
runs the validation engine and stores the issues and the review status."""
from datetime import datetime, timezone
from decimal import Decimal

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, ExtractedInformation, ValidationResult
from app.extraction.schema import specs
from app.services.similarity import hamming
from app.validation import engine
from app.validation.rules import Ctx, FieldView, Thresholds

NEAR_HASH_DISTANCE = 8  # coarse filter; the content check below decides
SAME_FIELDS_RATIO = 0.8
SAME_TEXT_RATIO = 97


def _norm(v: str | None) -> str:
    return "".join(ch for ch in (v or "").lower() if ch.isalnum())


def _text(doc: Document) -> str:
    return " ".join((p.ocr_text or "") for p in doc.pages)


def _same_content(doc: Document, other: Document) -> bool:
    """Two templated documents look alike (same layout) but carry different values, so a picture
    hash alone is not enough: the extracted field values (or, without fields, the text) must match."""
    mine = {f.field_name: _norm(f.field_value) for f in doc.fields if f.field_value}
    theirs = {f.field_name: _norm(f.field_value) for f in other.fields if f.field_value}
    shared = [k for k in mine if k in theirs]
    if len(shared) >= 3:
        return sum(mine[k] == theirs[k] for k in shared) / len(shared) >= SAME_FIELDS_RATIO
    a, b = _text(doc), _text(other)
    return bool(a.strip() and b.strip() and fuzz.ratio(a, b) >= SAME_TEXT_RATIO)


def _duplicates(db: Session, doc: Document) -> tuple[list[int], list[int]]:
    older = db.scalars(select(Document).where(Document.id < doc.id)).all()
    exact = [d.id for d in older if d.sha256 == doc.sha256]
    near: list[int] = []
    if doc.phash:
        for d in older:
            if d.id in exact or not d.phash or d.doc_category != doc.doc_category:
                continue
            if hamming(doc.phash, d.phash) <= NEAR_HASH_DISTANCE and _same_content(doc, d):
                near.append(d.id)
    return exact, near


def _policy_match(db: Session, doc: Document, number: str | None) -> int | None:
    if doc.doc_category != "claim_form" or not number:
        return None
    rows = db.execute(
        select(ExtractedInformation.document_id, ExtractedInformation.field_value)
        .join(Document, Document.id == ExtractedInformation.document_id)
        .where(ExtractedInformation.field_name == "policy_number", Document.doc_category == "insurance_policy", Document.id != doc.id)
    ).all()
    want = _norm(number)
    return next((i for i, v in rows if _norm(v) == want), None)


def build_ctx(db: Session, doc: Document) -> Ctx:
    confs = [p.avg_confidence for p in doc.pages if p.avg_confidence is not None]
    fields = {f.field_name: FieldView(f.field_name, f.field_value, f.confidence or 0.0, f.is_edited) for f in doc.fields}
    exact, near = _duplicates(db, doc)
    policy_number = fields["policy_number"].value if "policy_number" in fields else None
    return Ctx(
        category=doc.doc_category,
        specs=specs(doc.doc_category),
        fields=fields,
        today=datetime.now(timezone.utc).date(),
        has_text=any((p.ocr_text or "").strip() for p in doc.pages),
        ocr_confidence=sum(confs) / len(confs) if confs else None,
        class_confidence=doc.category_confidence,
        class_method=doc.classification_method,
        exact_duplicates=exact,
        near_duplicates=near,
        policy_match=_policy_match(db, doc, policy_number),
        limits=Thresholds(
            min_field_confidence=settings.validation_min_field_confidence,
            min_ocr_confidence=settings.validation_min_ocr_confidence,
            min_class_confidence=settings.validation_min_class_confidence,
            max_amount=Decimal(settings.validation_max_amount),
        ),
    )


def validate_document(db: Session, doc: Document) -> engine.Outcome | None:
    """Re-runs every rule and replaces the stored issues. Returns None (and changes nothing) when
    there is nothing to validate yet: no OCR text and no category."""
    ctx = build_ctx(db, doc)
    if not ctx.has_text and ctx.category is None:
        return None
    outcome = engine.validate(ctx)
    doc.issues.clear()
    db.flush()
    for i in outcome.issues:
        doc.issues.append(ValidationResult(rule_code=i.code, field_name=i.field, severity=i.severity, message=i.message))
    doc.review_status = outcome.status
    return outcome
