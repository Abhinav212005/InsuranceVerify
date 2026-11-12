from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import DocCategory, ProcessingStatus, ReviewStatus, Severity


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DocumentSummary(ORM):
    id: int
    filename: str
    file_type: str
    page_count: int
    status: ProcessingStatus
    review_status: ReviewStatus | None = None
    doc_category: DocCategory | None = None
    category_confidence: float | None = None
    uploaded_at: datetime
    processed_at: datetime | None = None
    processing_ms: int | None = None


class FieldOut(ORM):
    field_name: str
    field_value: str | None = None
    original_value: str | None = None
    confidence: float = 0.0
    source_page: int | None = None
    bbox_json: dict | list | None = None  # {"bbox": [x,y,w,h] fractions, "method", "source", "note"}
    is_edited: bool = False


class IssueOut(ORM):
    rule_code: str
    field_name: str | None = None
    severity: Severity
    message: str


class PageOut(ORM):
    page_number: int
    recipe: str | None = None
    stage_paths_json: dict | None = None
    quality_before_json: dict | None = None
    quality_after_json: dict | None = None
    skew_angle: float | None = None
    ocr_text: str | None = None
    ocr_words_json: list | None = None
    avg_confidence: float | None = None
    low_conf_count: int | None = None
    psm: int | None = None
    ocr_source: str | None = None


class DocumentDetail(DocumentSummary):
    classification_method: str | None = None
    classification_json: dict | None = None
    error_message: str | None = None
    current_stage: str | None = None
    pages: list[PageOut] = []
    fields: list[FieldOut] = []
    issues: list[IssueOut] = []


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int
    size: int


class UploadedItem(BaseModel):
    id: int
    filename: str
    status: ProcessingStatus
    page_count: int
    duplicate_of: int | None = None  # id of an earlier identical upload


class RejectedItem(BaseModel):
    filename: str
    code: str
    message: str


class UploadResponse(BaseModel):
    items: list[UploadedItem]
    rejected: list[RejectedItem] = []


class StatusOut(BaseModel):
    status: ProcessingStatus
    stage: str | None = None
    progress: int = Field(ge=0, le=100)


class FieldsUpdate(BaseModel):
    fields: dict[str, str | None] = Field(default_factory=dict)
    category: DocCategory | None = None


class DistributionItem(BaseModel):
    type: str
    count: int


class DayStat(BaseModel):
    day: str
    uploads: int
    avg_ms: int | None = None


class StatsSummary(BaseModel):
    total: int
    valid: int
    needs_review: int
    incomplete: int
    failed: int
    avg_processing_ms: int | None = None
    distribution: list[DistributionItem]
    per_day: list[DayStat]
