from enum import Enum


class DocCategory(str, Enum):
    insurance_policy = "insurance_policy"
    claim_form = "claim_form"
    identity_document = "identity_document"
    medical_bill = "medical_bill"
    repair_invoice = "repair_invoice"
    other = "other"


class ProcessingStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    ingested = "ingested"  # pages rendered; analysis stages are added in later phases
    processed = "processed"
    failed = "failed"


class ReviewStatus(str, Enum):
    valid = "valid"
    needs_review = "needs_review"
    incomplete = "incomplete"


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"
