from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    file_type: Mapped[str] = mapped_column(String(10))  # pdf / png / jpg
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="uploaded", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(30), nullable=True)
    review_status: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    doc_category: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    category_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_method: Mapped[str | None] = mapped_column(String(20), nullable=True)
    classification_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # probabilities, evidence
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    pages: Mapped[list["ProcessingResult"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ProcessingResult.page_number"
    )
    fields: Mapped[list["ExtractedInformation"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ExtractedInformation.id"
    )
    issues: Mapped[list["ValidationResult"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="ValidationResult.id"
    )


class ProcessingResult(Base):
    """One row per document page: DIP stages, quality metrics and OCR output."""

    __tablename__ = "processing_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    recipe: Mapped[str | None] = mapped_column(String(40), nullable=True)
    params_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stage_paths_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_before_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_after_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    skew_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_words_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_conf_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    psm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ocr_source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # ocr / text_layer
    preprocess_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ocr_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped[Document] = relationship(back_populates="pages")


class ExtractedInformation(Base):
    """One row per extracted field."""

    __tablename__ = "extracted_information"
    __table_args__ = (UniqueConstraint("document_id", "field_name", name="uq_doc_field"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(60))
    field_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    document: Mapped[Document] = relationship(back_populates="fields")


class ValidationResult(Base):
    """One row per validation issue."""

    __tablename__ = "validation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    rule_code: Mapped[str] = mapped_column(String(50))
    field_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    severity: Mapped[str] = mapped_column(String(10))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="issues")
