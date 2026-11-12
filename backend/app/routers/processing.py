from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.core.errors import AppError, NotFoundError
from app.dip import pipeline
from app.repositories.document_repo import DocumentRepo
from app.routers.documents import get_repo
from app.core.config import settings
from app.services import classification_service, dip_service, extraction_service, ocr_service, validation_service

router = APIRouter(tags=["processing"])


class PreprocessRequest(BaseModel):
    recipe: str = pipeline.DEFAULT_RECIPE
    # per-step parameter overrides, e.g. {"threshold": {"method": "otsu"}}
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    pages: list[int] | None = None  # default: all pages


def _doc(repo: DocumentRepo, doc_id: int):
    doc = repo.get(doc_id)
    if doc is None:
        raise NotFoundError("Document")
    return doc


@router.get("/pipeline/recipes")
def list_recipes():
    """Available preprocessing recipes with the steps, OpenCV functions and parameters."""
    return {"default": pipeline.DEFAULT_RECIPE, "recipes": pipeline.recipe_catalog()}


@router.post("/documents/{doc_id}/preprocess")
def preprocess(doc_id: int, body: PreprocessRequest, repo: DocumentRepo = Depends(get_repo)):
    """Runs a recipe on the document's pages (synchronously) and returns stage images,
    metrics and step timings. The result becomes the page's active recipe."""
    doc = _doc(repo, doc_id)
    if doc.status not in ("ingested", "processed"):
        raise AppError("NOT_READY", "The document's pages are not rendered yet", 409)
    results = dip_service.preprocess_document(doc, body.recipe, body.params, body.pages)
    repo.db.commit()
    return {"recipe": body.recipe, "pages": results}


class ClassifyRequest(BaseModel):
    method: Literal["keyword", "ml", "hybrid"] | None = None  # default from settings


@router.post("/documents/{doc_id}/classify")
def classify_document(doc_id: int, body: ClassifyRequest, repo: DocumentRepo = Depends(get_repo)):
    """(Re)classifies the document from its OCR text. Also replaces a manual category choice."""
    doc = _doc(repo, doc_id)
    result = classification_service.classify_document(doc, body.method)
    extraction_service.extract_document(doc)  # fields depend on the category
    validation_service.validate_document(repo.db, doc)
    repo.db.commit()
    return result


@router.post("/documents/{doc_id}/validate")
def validate_document(doc_id: int, repo: DocumentRepo = Depends(get_repo)):
    """Re-runs all validation rules. The status says whether the data is complete, never whether a claim is approved."""
    doc = _doc(repo, doc_id)
    outcome = validation_service.validate_document(repo.db, doc)
    if outcome is None:
        raise AppError("NOTHING_TO_VALIDATE", "The document has no OCR text or category yet.", 409)
    repo.db.commit()
    counts = {s: sum(i.severity == s for i in outcome.issues) for s in ("error", "warning", "info")}
    return {
        "status": outcome.status,
        "counts": counts,
        "issues": [{"rule_code": i.code, "severity": i.severity, "field_name": i.field, "message": i.message} for i in outcome.issues],
        "disclaimer": "This status reflects data completeness only. It is not a claim decision.",
    }


class ExtractRequest(BaseModel):
    category: str | None = None  # default: the document's current category


@router.post("/documents/{doc_id}/extract")
def extract_fields(doc_id: int, body: ExtractRequest, repo: DocumentRepo = Depends(get_repo)):
    """(Re)extracts structured fields. Values a person edited are kept."""
    from app.extraction.schema import FIELDS

    doc = _doc(repo, doc_id)
    category = body.category or doc.doc_category
    if body.category is not None and body.category not in FIELDS:
        raise AppError("INVALID_CATEGORY", f"category must be one of {list(FIELDS)}", 422)
    if body.category and body.category != doc.doc_category:
        doc.doc_category = body.category
        doc.classification_method = "manual"
        doc.category_confidence = 1.0
    fields = extraction_service.extract_document(doc, category)
    validation_service.validate_document(repo.db, doc)
    repo.db.commit()
    return {"category": category, "fields": [
        {"field_name": f.field_name, "field_value": f.field_value, "confidence": f.confidence,
         "source_page": f.source_page, "bbox_json": f.bbox_json, "is_edited": f.is_edited} for f in fields]}


@router.get("/extraction/schema")
def extraction_schema():
    """Fields extracted for each document category (label, kind, mandatory)."""
    from app.extraction.schema import schema_payload

    return schema_payload()


@router.get("/classification/info")
def classification_info():
    """Categories and which classification methods are available right now."""
    from app.classification import classifier, ml_classifier
    from app.classification.keywords import CATEGORIES

    return {
        "categories": CATEGORIES,
        "methods": classifier.available_methods(),
        "default_method": settings.classifier_method,
        "ml_model": ml_classifier.metadata(),
    }


class OcrRequest(BaseModel):
    psm: int | None = None  # page segmentation mode; default from settings
    source: Literal["processed", "original"] = "processed"  # which image to read
    recipe: str | None = None  # which preprocessing recipe's output (default: the page's active one)
    pages: list[int] | None = None
    force: bool = False  # re-OCR digital-PDF pages instead of keeping their text layer


@router.post("/documents/{doc_id}/ocr")
def run_ocr(doc_id: int, body: OcrRequest, repo: DocumentRepo = Depends(get_repo)):
    """Runs Tesseract (synchronously) on the chosen pages and stores the result."""
    doc = _doc(repo, doc_id)
    if doc.status not in ("ingested", "processed"):
        raise AppError("NOT_READY", "The document's pages are not rendered yet", 409)
    if body.recipe is not None and body.recipe not in pipeline.RECIPES:
        raise AppError("UNKNOWN_RECIPE", f"Unknown recipe '{body.recipe}'", 422)
    ocr_service.ocr_document(doc, body.psm, body.source, body.recipe, body.pages, body.force)
    repo.db.commit()
    return ocr_service.document_payload(doc)


@router.get("/documents/{doc_id}/ocr")
def get_ocr(
    doc_id: int,
    threshold: float | None = Query(None, ge=0, le=100, description="Words below this confidence count as low"),
    repo: DocumentRepo = Depends(get_repo),
):
    return ocr_service.document_payload(_doc(repo, doc_id), threshold)


@router.get("/documents/{doc_id}/pages/{page_number}/stages")
def page_stages(
    doc_id: int,
    page_number: int,
    recipe: str | None = Query(None, description="Defaults to the page's active recipe"),
    repo: DocumentRepo = Depends(get_repo),
):
    return dip_service.read_stages(_doc(repo, doc_id), page_number, recipe)
