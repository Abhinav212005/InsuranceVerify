from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Document, ExtractedInformation, utcnow


class DocumentRepo:
    def __init__(self, db: Session):
        self.db = db

    # ---- create / read -------------------------------------------------
    def add(self, doc: Document) -> Document:
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get(self, doc_id: int) -> Document | None:
        stmt = (
            select(Document)
            .where(Document.id == doc_id)
            .options(
                selectinload(Document.pages),
                selectinload(Document.fields),
                selectinload(Document.issues),
            )
        )
        return self.db.scalars(stmt).first()

    def find_by_sha256(self, sha: str) -> Document | None:
        return self.db.scalars(select(Document).where(Document.sha256 == sha)).first()

    def list(
        self,
        q: str | None = None,
        category: str | None = None,
        review_status: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[Document], int]:
        stmt = select(Document)
        if q:
            like = f"%{q}%"
            matching_values = select(ExtractedInformation.document_id).where(
                ExtractedInformation.field_value.ilike(like)
            )
            stmt = stmt.where(Document.filename.ilike(like) | Document.id.in_(matching_values))
        if category:
            stmt = stmt.where(Document.doc_category == category)
        if review_status:
            stmt = stmt.where(Document.review_status == review_status)
        if date_from:
            stmt = stmt.where(
                Document.uploaded_at >= datetime.combine(date_from, time.min, tzinfo=timezone.utc)
            )
        if date_to:
            stmt = stmt.where(
                Document.uploaded_at < datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
            )
        total = self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        stmt = stmt.order_by(Document.uploaded_at.desc(), Document.id.desc())
        stmt = stmt.offset((page - 1) * size).limit(size)
        return list(self.db.scalars(stmt)), total

    # ---- update / delete ----------------------------------------------
    def delete(self, doc: Document) -> None:
        self.db.delete(doc)  # cascades to pages, fields, issues
        self.db.commit()

    def update_fields(self, doc: Document, values: dict[str, str | None]) -> None:
        existing = {f.field_name: f for f in doc.fields}
        for name, value in values.items():
            value = value.strip() if isinstance(value, str) else value
            value = value or None
            row = existing.get(name)
            if row is None:
                doc.fields.append(
                    ExtractedInformation(
                        field_name=name,
                        field_value=value,
                        original_value=None,
                        confidence=1.0,  # entered by a human
                        is_edited=True,
                    )
                )
            else:
                # An explicit update is a human decision, even when the value is unchanged: confirming
                # a correct but low-confidence OCR reading is how a reviewer clears its warning.
                row.field_value = value
                row.is_edited = True
                row.confidence = 1.0
                row.updated_at = utcnow()
        self.db.commit()
        self.db.refresh(doc)

    def set_category(self, doc: Document, category: str) -> None:
        doc.doc_category = category
        doc.classification_method = "manual"
        doc.category_confidence = 1.0
        self.db.commit()

    # ---- statistics ----------------------------------------------------
    def stats(self, days: int = 7) -> dict:
        db = self.db
        total = db.scalar(select(func.count(Document.id))) or 0
        by_review = dict(
            db.execute(select(Document.review_status, func.count()).group_by(Document.review_status)).all()
        )
        failed = db.scalar(select(func.count(Document.id)).where(Document.status == "failed")) or 0
        avg_ms = db.scalar(select(func.avg(Document.processing_ms)))
        dist = db.execute(
            select(Document.doc_category, func.count())
            .where(Document.doc_category.is_not(None))
            .group_by(Document.doc_category)
            .order_by(func.count().desc())
        ).all()

        start = datetime.combine(datetime.now(timezone.utc).date() - timedelta(days=days - 1), time.min, tzinfo=timezone.utc)
        rows = db.execute(
            select(Document.uploaded_at, Document.processing_ms).where(Document.uploaded_at >= start)
        ).all()
        buckets: dict[str, list] = {
            (start.date() + timedelta(days=i)).isoformat(): [] for i in range(days)
        }
        for uploaded_at, ms in rows:
            key = uploaded_at.date().isoformat()
            if key in buckets:
                buckets[key].append(ms)
        per_day = []
        for day, vals in buckets.items():
            done = [v for v in vals if v is not None]
            per_day.append(
                {"day": day, "uploads": len(vals), "avg_ms": int(sum(done) / len(done)) if done else None}
            )
        return {
            "total": total,
            "valid": by_review.get("valid", 0),
            "needs_review": by_review.get("needs_review", 0),
            "incomplete": by_review.get("incomplete", 0),
            "failed": failed,
            "avg_processing_ms": int(avg_ms) if avg_ms is not None else None,
            "distribution": [{"type": c, "count": n} for c, n in dist],
            "per_day": per_day,
        }
