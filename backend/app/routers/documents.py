import shutil
from datetime import date
from pathlib import Path

import json
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Response, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.enums import DocCategory, ReviewStatus
from app.core.errors import AppError, NotFoundError
from app.db.session import SessionLocal, get_db
from app.repositories.document_repo import DocumentRepo
from app.schemas.document import (
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    FieldsUpdate,
    RejectedItem,
    StatusOut,
    UploadedItem,
    UploadResponse,
)
from app.services import extraction_service, pipeline_service, report_service, upload_service, validation_service

router = APIRouter(prefix="/documents", tags=["documents"])

STAGES = [
    "uploaded", "rendering", "rendered", "preprocessing", "preprocessed", "ocr", "ocr_done",
    "classifying", "classified", "extracting", "extracted", "validating", "done",
]


def get_repo(db: Session = Depends(get_db)) -> DocumentRepo:
    return DocumentRepo(db)


def _get_or_404(repo: DocumentRepo, doc_id: int):
    doc = repo.get(doc_id)
    if doc is None:
        raise NotFoundError("Document")
    return doc


def _revalidate(repo: DocumentRepo, doc) -> None:
    validation_service.validate_document(repo.db, doc)
    repo.db.commit()


@router.get("", response_model=DocumentList)
def list_documents(
    q: str | None = Query(None, max_length=100, description="Filename or extracted-field text"),
    type: DocCategory | None = None,
    status: ReviewStatus | None = None,
    date_from: date | None = Query(None, alias="from"),
    date_to: date | None = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    repo: DocumentRepo = Depends(get_repo),
):
    items, total = repo.list(
        q=q,
        category=type.value if type else None,
        review_status=status.value if status else None,
        date_from=date_from,
        date_to=date_to,
        page=page,
        size=size,
    )
    return DocumentList(
        items=[DocumentSummary.model_validate(d) for d in items], total=total, page=page, size=size
    )


def get_session_factory():
    return SessionLocal  # overridden in tests


@router.post("", response_model=UploadResponse, status_code=201)
def upload_documents(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    session_factory=Depends(get_session_factory),
):
    """Accepts one or more files. Each is validated independently; valid ones are stored
    and processed in the background (poll GET /documents/{id}/status)."""
    accepted: list[UploadedItem] = []
    rejected: list[RejectedItem] = []
    for f in files:
        name = f.filename or "unnamed"
        try:
            data = upload_service.read_limited(f)
            doc, dup = pipeline_service.create_document(db, name, data)
        except AppError as e:
            rejected.append(RejectedItem(filename=name, code=e.code, message=e.message))
            continue
        accepted.append(
            UploadedItem(id=doc.id, filename=doc.filename, status=doc.status, page_count=doc.page_count, duplicate_of=dup)
        )
        background.add_task(pipeline_service.ingest_document, doc.id, session_factory)
    if not accepted:
        first = rejected[0]
        raise AppError(first.code, first.message, 413 if first.code == "FILE_TOO_LARGE" else 400)
    return UploadResponse(items=accepted, rejected=rejected)


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: int, repo: DocumentRepo = Depends(get_repo)):
    return _get_or_404(repo, doc_id)


@router.post("/{doc_id}/process", status_code=202)
def reprocess_document(
    doc_id: int,
    background: BackgroundTasks,
    repo: DocumentRepo = Depends(get_repo),
    session_factory=Depends(get_session_factory),
):
    """Runs the whole pipeline again from the stored original (e.g. after a failure).
    Poll GET /documents/{id}/status for progress."""
    doc = _get_or_404(repo, doc_id)
    if doc.status == "processing":
        raise AppError("ALREADY_PROCESSING", "This document is already being processed.", 409)
    doc.status, doc.current_stage, doc.error_message = "processing", "uploaded", None
    repo.db.commit()
    background.add_task(pipeline_service.ingest_document, doc.id, session_factory)
    return {"id": doc.id, "status": doc.status}


@router.get("/{doc_id}/report")
def download_report(doc_id: int, format: Literal["json", "pdf"] = "json", repo: DocumentRepo = Depends(get_repo)):
    """Downloadable summary: fields, issues, recognised text and the completeness disclaimer."""
    report = report_service.build_report(_get_or_404(repo, doc_id))
    name = f"document_{doc_id}_report"
    if format == "pdf":
        return Response(
            report_service.to_pdf(report), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )
    return Response(
        json.dumps(report, indent=2), media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
    )


@router.get("/{doc_id}/status", response_model=StatusOut)
def get_status(doc_id: int, repo: DocumentRepo = Depends(get_repo)):
    doc = _get_or_404(repo, doc_id)
    stage = doc.current_stage or ("done" if doc.status == "processed" else "uploaded")
    progress = round(STAGES.index(stage) / (len(STAGES) - 1) * 100) if stage in STAGES else 0
    return StatusOut(status=doc.status, stage=stage, progress=progress)


@router.patch("/{doc_id}/fields", response_model=DocumentDetail)
def update_fields(doc_id: int, body: FieldsUpdate, repo: DocumentRepo = Depends(get_repo)):
    doc = _get_or_404(repo, doc_id)
    if body.fields:
        repo.update_fields(doc, body.fields)
    if body.category and body.category.value != doc.doc_category:
        repo.set_category(doc, body.category.value)
        try:  # a different type has different fields; edited values are preserved
            extraction_service.extract_document(doc)
            repo.db.commit()
        except AppError:
            pass  # no OCR text yet: nothing to extract
    elif body.category:
        repo.set_category(doc, body.category.value)
    _revalidate(repo, doc)  # corrections can fix (or introduce) problems
    return _get_or_404(repo, doc_id)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int, repo: DocumentRepo = Depends(get_repo)):
    doc = _get_or_404(repo, doc_id)
    repo.delete(doc)
    for sub in ("uploads", "processed"):
        target = (settings.storage_dir / sub / str(doc_id)).resolve()
        root = (settings.storage_dir / sub).resolve()
        if root in target.parents and Path(target).exists():  # never delete outside storage
            shutil.rmtree(target, ignore_errors=True)
    return Response(status_code=204)
