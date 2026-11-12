"""Orchestrates document processing: ingestion (file -> page images), OpenCV
preprocessing, OCR, classification, field extraction and validation."""
import logging
import time
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppError
from app.db.models import Document, ProcessingResult, utcnow
from app.repositories.document_repo import DocumentRepo
from app.dip import pipeline as dip_pipeline
from app.services import (
    classification_service, dip_service, extraction_service, ocr_service, pdf_service, similarity,
    upload_service, validation_service,
)

log = logging.getLogger(__name__)


def create_document(db: Session, filename: str, data: bytes) -> tuple[Document, int | None]:
    """Validate a single upload, store it and create the DB row.
    Returns (document, id of an earlier identical upload or None)."""
    filename = upload_service.safe_filename(filename)
    file_type = upload_service.detect_type(filename, data[:16])
    page_count = pdf_service.inspect_pdf(data) if file_type == "pdf" else (pdf_service.inspect_image(data) or 1)

    repo = DocumentRepo(db)
    sha = upload_service.sha256_hex(data)
    earlier = repo.find_by_sha256(sha)

    doc = repo.add(
        Document(
            filename=filename,
            stored_path="",
            file_type=file_type,
            file_size=len(data),
            sha256=sha,
            page_count=page_count,
            status="uploaded",
            current_stage="uploaded",
        )
    )
    doc.stored_path = upload_service.save_original(doc.id, file_type, data)
    db.commit()
    return doc, earlier.id if earlier else None


def ingest_document(doc_id: int, session_factory: Callable[[], Session]) -> None:
    """Background job: render pages to PNG and store per-page results."""
    db = session_factory()
    started = time.perf_counter()
    try:
        repo = DocumentRepo(db)
        doc = repo.get(doc_id)
        if doc is None:
            return
        doc.status, doc.current_stage, doc.error_message = "processing", "rendering", None
        db.commit()

        src = settings.storage_dir / doc.stored_path
        pages = (
            pdf_service.render_pdf(doc.id, src) if doc.file_type == "pdf" else pdf_service.render_image(doc.id, src)
        )

        doc.pages.clear()
        db.flush()
        for p in pages:
            base = f"/files/{doc.id}/page_{p.page_number}"
            doc.pages.append(
                ProcessingResult(
                    page_number=p.page_number,
                    stage_paths_json={"original": f"{base}/original.png", "thumb": f"{base}/thumb.jpg"},
                    # Digital PDFs already contain text, so OCR can be skipped for those pages.
                    ocr_text=p.text,
                    ocr_source="text_layer" if p.text else None,
                    avg_confidence=100.0 if p.text else None,
                )
            )
        doc.page_count = len(pages)
        doc.phash = similarity.dhash(dip_service._load_original(doc.id, 1))  # for near-duplicate detection
        doc.current_stage = "preprocessing"
        db.commit()

        # Digital image processing with the default recipe (users can re-run others in the Studio).
        for row in doc.pages:
            dip_service.preprocess_page(doc, row, dip_pipeline.DEFAULT_RECIPE, None)

        doc.current_stage = "preprocessed"
        if settings.auto_ocr:
            doc.current_stage = "ocr"
            db.commit()
            try:
                for row in doc.pages:
                    ocr_service.ocr_page(doc, row)
                doc.current_stage = "ocr_done"
            except AppError as e:
                # Keep the preprocessing results; OCR can be re-run from the API/UI.
                log.warning("OCR failed for document %s: %s", doc.id, e.message)
                doc.current_stage, doc.error_message = "ocr_failed", f"OCR failed: {e.message}"[:500]

        if settings.auto_classify and doc.current_stage == "ocr_done" and classification_service.document_text(doc).strip():
            doc.current_stage = "classifying"
            db.commit()
            classification_service.classify_document(doc)
            doc.current_stage = "classified"
            if settings.auto_extract:
                doc.current_stage = "extracting"
                db.commit()
                extraction_service.extract_document(doc)
                doc.current_stage = "extracted"
                if settings.auto_validate:
                    doc.current_stage = "validating"
                    db.commit()
                    validation_service.validate_document(db, doc)
                    doc.current_stage = "done"

        # "processed" = every enabled stage ran, including validation; otherwise ready for the remaining steps
        doc.status = "processed" if doc.current_stage == "done" else "ingested"
        doc.processing_ms = int((time.perf_counter() - started) * 1000)
        doc.processed_at = utcnow()
        db.commit()
        log.info("document %s ingested: %d page(s) in %d ms", doc.id, len(pages), doc.processing_ms)
    except Exception as exc:  # never let a background job die silently
        log.exception("ingestion failed for document %s", doc_id)
        db.rollback()
        doc = db.get(Document, doc_id)
        if doc:
            doc.status, doc.error_message = "failed", f"{type(exc).__name__}: {exc}"[:500]
            db.commit()
    finally:
        db.close()


def stored_file(doc: Document) -> Path:
    return settings.storage_dir / doc.stored_path
