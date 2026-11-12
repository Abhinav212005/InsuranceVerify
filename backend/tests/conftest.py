from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.db.models import Document, ExtractedInformation, ProcessingResult, ValidationResult
from app.core.config import settings
from app.db.session import create_db_engine, get_db, init_db
from app.main import app
from app.routers.documents import get_session_factory


@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    """Every test writes files to its own temp dir, never to the real storage folder."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "auto_ocr", False)  # OCR tests switch it on explicitly (it is slow)
    (tmp_path / "storage" / "uploads").mkdir(parents=True)
    (tmp_path / "storage" / "processed").mkdir(parents=True)
    return tmp_path / "storage"


@pytest.fixture()
def session_factory():
    engine = create_db_engine("sqlite://")  # isolated in-memory DB per test
    init_db(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture()
def db(session_factory):
    s = session_factory()
    yield s
    s.close()


@pytest.fixture()
def client(session_factory):
    def _override():
        s = session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    yield TestClient(app)  # no `with`: skip lifespan so the real DB is never touched
    app.dependency_overrides.clear()


def make_doc(db, filename="policy_1.pdf", category="insurance_policy", review="valid", days_ago=0, **kw):
    doc = Document(
        filename=filename,
        stored_path=f"uploads/{filename}",
        file_type="pdf",
        file_size=1234,
        sha256=kw.pop("sha256", filename.ljust(64, "0")),
        page_count=kw.pop("page_count", 1),
        status=kw.pop("status", "processed"),
        review_status=review,
        doc_category=category,
        category_confidence=0.9,
        uploaded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        processing_ms=kw.pop("processing_ms", 4000),
        **kw,
    )
    doc.pages.append(ProcessingResult(page_number=1, ocr_text="hello", avg_confidence=88.0))
    doc.fields.append(
        ExtractedInformation(
            field_name="policy_number", field_value="POL-1", original_value="POL-1", confidence=0.9
        )
    )
    doc.issues.append(
        ValidationResult(rule_code="LOW_CONFIDENCE_FIELD", field_name="policy_number", severity="warning", message="m")
    )
    db.add(doc)
    db.commit()
    return doc
