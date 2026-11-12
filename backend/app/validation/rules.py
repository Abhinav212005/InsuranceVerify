"""Validation rules. Each rule is a small pure function: Ctx in, list of Issue out.

Severity says how serious the finding is; the document STATUS (decided in engine.py) says what
the reviewer must do:
  Incomplete    - a mandatory field is missing
  Needs Review  - nothing mandatory is missing, but something looks wrong or uncertain
  Valid         - complete and no warnings ("data complete", never "approved")
"""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from app.extraction.schema import FieldSpec

MISSING_CODES = {"policy_number": "POLICY_NUMBER_MISSING", "claim_number": "CLAIM_NUMBER_MISSING"}


@dataclass
class Issue:
    code: str
    severity: str  # error | warning | info
    message: str
    field: str | None = None


@dataclass
class FieldView:
    name: str
    value: str | None
    confidence: float
    edited: bool = False


@dataclass
class Thresholds:
    min_field_confidence: float = 0.6
    min_ocr_confidence: float = 60.0
    min_class_confidence: float = 0.5
    max_amount: Decimal = Decimal("1000000000")
    max_age_years: int = 110


@dataclass
class Ctx:
    category: str | None
    specs: list[FieldSpec]
    fields: dict[str, FieldView]
    today: date
    has_text: bool = True
    ocr_confidence: float | None = None  # average over pages, 0-100
    class_confidence: float | None = None
    class_method: str | None = None
    exact_duplicates: list[int] = field(default_factory=list)
    near_duplicates: list[int] = field(default_factory=list)
    policy_match: int | None = None  # id of an uploaded policy document with the claim's policy number
    limits: Thresholds = field(default_factory=Thresholds)

    def value(self, name: str) -> str | None:
        f = self.fields.get(name)
        v = (f.value or "").strip() if f else ""
        return v or None


def _label(ctx: Ctx, name: str) -> str:
    return next((s.label for s in ctx.specs if s.name == name), name)


def _pretty(d: date) -> str:
    return d.strftime("%d %b %Y").lstrip("0")


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


# ---- field rules ---------------------------------------------------------------------------------
def missing_fields(ctx: Ctx) -> list[Issue]:
    out = []
    for s in ctx.specs:
        if ctx.value(s.name):
            continue
        if s.mandatory:
            out.append(Issue(MISSING_CODES.get(s.name, "MISSING_FIELD"), "error",
                             f"{s.label} is missing. It is required for this document type.", s.name))
        else:
            out.append(Issue("OPTIONAL_MISSING", "info", f"{s.label} was not found (optional).", s.name))
    return out


def date_rules(ctx: Ctx) -> list[Issue]:
    out: list[Issue] = []
    parsed: dict[str, date] = {}
    for s in ctx.specs:
        v = ctx.value(s.name)
        if s.kind != "date" or not v:
            continue
        d = _parse_date(v)
        if d is None or d.year < 1900 or d.year > 2100:
            out.append(Issue("INVALID_DATE", "warning",
                             f"{s.label} “{v}” is not a valid date (expected a real calendar date).", s.name))
            continue
        parsed[s.name] = d

    end, start = parsed.get("policy_end_date"), parsed.get("policy_start_date")
    if end and end < ctx.today:
        out.append(Issue("POLICY_EXPIRED", "warning", f"The policy expired on {_pretty(end)}.", "policy_end_date"))
    if start and end and end < start:
        out.append(Issue("DATE_ORDER", "error", "The policy expiry date is before its start date.", "policy_end_date"))
    for name in ("incident_date", "invoice_date", "repair_date"):
        d = parsed.get(name)
        if d and d > ctx.today:
            out.append(Issue("FUTURE_DATE", "warning", f"{_label(ctx, name)} ({_pretty(d)}) is in the future.", name))
    dob = parsed.get("date_of_birth")
    if dob:
        age = (ctx.today - dob).days / 365.25
        if dob > ctx.today:
            out.append(Issue("INVALID_DATE", "warning", "Date of birth is in the future.", "date_of_birth"))
        elif age > ctx.limits.max_age_years:
            out.append(Issue("INVALID_DATE", "warning", f"Date of birth implies an age of {age:.0f}.", "date_of_birth"))
    exp = parsed.get("expiry_date")
    if exp and exp < ctx.today:
        out.append(Issue("ID_EXPIRED", "warning", f"This identity document expired on {_pretty(exp)}.", "expiry_date"))
    return out


def amount_rules(ctx: Ctx) -> list[Issue]:
    out = []
    for s in ctx.specs:
        v = ctx.value(s.name)
        if s.kind != "amount" or not v:
            continue
        try:
            amount = Decimal(v.replace(",", ""))
        except InvalidOperation:
            out.append(Issue("INVALID_AMOUNT", "warning", f"{s.label} “{v}” is not a valid amount.", s.name))
            continue
        if amount <= 0:
            out.append(Issue("INVALID_AMOUNT", "warning", f"{s.label} must be greater than zero (found {v}).", s.name))
        elif amount > ctx.limits.max_amount:
            out.append(Issue("INVALID_AMOUNT", "warning", f"{s.label} {amount:,.0f} is unusually large. Please check for a misread digit.", s.name))
    return out


def field_confidence(ctx: Ctx) -> list[Issue]:
    out = []
    for s in ctx.specs:
        f = ctx.fields.get(s.name)
        if f and ctx.value(s.name) and not f.edited and f.confidence < ctx.limits.min_field_confidence:
            out.append(Issue("LOW_CONFIDENCE_FIELD", "warning",
                             f"{s.label} was read with low confidence ({f.confidence:.0%}). Please verify it against the document.", s.name))
    return out


# ---- document-level rules -----------------------------------------------------------------------------
def ocr_quality(ctx: Ctx) -> list[Issue]:
    if not ctx.has_text:
        return [Issue("NO_TEXT_FOUND", "warning", "No text could be read from this document. The image may be blank or too poor in quality.")]
    if ctx.ocr_confidence is not None and ctx.ocr_confidence < ctx.limits.min_ocr_confidence:
        return [Issue("LOW_OCR_CONFIDENCE", "warning",
                      f"Overall OCR confidence is low ({ctx.ocr_confidence:.0f}%). Check the extracted values against the original.")]
    return []


def classification_quality(ctx: Ctx) -> list[Issue]:
    if ctx.category is None:
        return [Issue("NOT_CLASSIFIED", "warning", "The document type has not been determined yet.")]
    if ctx.class_method != "manual" and ctx.class_confidence is not None and ctx.class_confidence < ctx.limits.min_class_confidence:
        return [Issue("LOW_CLASSIFICATION_CONFIDENCE", "warning",
                      f"The document type was predicted with low confidence ({ctx.class_confidence:.0%}). Please confirm it.")]
    return []


def duplicates(ctx: Ctx) -> list[Issue]:
    out = []
    if ctx.exact_duplicates:
        ids = ", ".join(f"#{i}" for i in ctx.exact_duplicates)
        out.append(Issue("DUPLICATE_DOCUMENT", "warning", f"This file is identical to an earlier upload ({ids})."))
    elif ctx.near_duplicates:
        ids = ", ".join(f"#{i}" for i in ctx.near_duplicates)
        out.append(Issue("NEAR_DUPLICATE_DOCUMENT", "warning", f"This document looks almost identical to an earlier upload ({ids}); it may be a re-scan."))
    return out


def policy_crosscheck(ctx: Ctx) -> list[Issue]:
    num = ctx.value("policy_number")
    if ctx.category != "claim_form" or not num:
        return []
    if ctx.policy_match:
        return [Issue("POLICY_MATCHED", "info", f"The policy number matches an uploaded policy document (#{ctx.policy_match}).", "policy_number")]
    return [Issue("POLICY_NOT_FOUND", "info", "No uploaded policy document has this policy number.", "policy_number")]


RULES = [missing_fields, date_rules, amount_rules, field_confidence, ocr_quality, classification_quality, duplicates, policy_crosscheck]
