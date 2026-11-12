from datetime import date
from decimal import Decimal

import cv2
import pytest

from app.core.config import settings
from app.extraction.schema import specs
from app.services.similarity import dhash, hamming
from app.validation import engine
from app.validation.rules import Ctx, FieldView
from tests.dip_helpers import add_gaussian_noise, make_page

P = "/api/v1"
TODAY = date(2026, 6, 1)


def ctx_for(category, values, *, conf=0.9, edited=(), **kw):
    """Build a validation context from {field: value}; every spec'd field absent from values is missing."""
    fields = {k: FieldView(k, v, conf, k in edited) for k, v in values.items()}
    base = dict(category=category, specs=specs(category), fields=fields, today=TODAY, ocr_confidence=90.0, class_confidence=0.9)
    base.update(kw)
    return Ctx(**base)


GOOD_POLICY = {
    "customer_name": "Aarav Sharma", "policy_number": "POL-2025-001234", "insurance_company": "Northstar Insurance",
    "policy_start_date": "2025-04-01", "policy_end_date": "2027-03-31", "premium_amount": "12450.00", "coverage_amount": "500000.00",
}


def codes(outcome):
    return [i.code for i in outcome.issues]


# ---- status logic --------------------------------------------------------------------------------
def test_complete_clean_document_is_valid():
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY))
    assert out.status == "valid" and out.issues == []


def test_missing_mandatory_field_is_incomplete_with_specific_codes():
    values = {k: v for k, v in GOOD_POLICY.items() if k not in ("policy_number", "premium_amount")}
    out = engine.validate(ctx_for("insurance_policy", values))
    assert out.status == "incomplete"
    assert {"POLICY_NUMBER_MISSING", "MISSING_FIELD"} <= set(codes(out))
    missing = [i for i in out.issues if i.severity == "error"]
    assert {i.field for i in missing} == {"policy_number", "premium_amount"}


def test_claim_number_missing_code():
    out = engine.validate(ctx_for("claim_form", {"policy_number": "P-1", "customer_name": "A B", "incident_date": "2026-01-01", "claim_amount": "100.00"}))
    assert "CLAIM_NUMBER_MISSING" in codes(out) and out.status == "incomplete"


def test_missing_optional_field_is_only_info():
    values = {k: v for k, v in GOOD_POLICY.items() if k != "coverage_amount"}
    out = engine.validate(ctx_for("insurance_policy", values))
    assert out.status == "valid" and codes(out) == ["OPTIONAL_MISSING"] and out.issues[0].severity == "info"


def test_incomplete_beats_needs_review():
    out = engine.validate(ctx_for("insurance_policy", {**{k: v for k, v in GOOD_POLICY.items() if k != "customer_name"}, "policy_end_date": "2020-01-01"}))
    assert out.status == "incomplete" and "POLICY_EXPIRED" in codes(out)


def test_issues_sorted_errors_first():
    out = engine.validate(ctx_for("insurance_policy", {**{k: v for k, v in GOOD_POLICY.items() if k != "policy_number"}, "policy_end_date": "2020-01-01"}))
    sev = [i.severity for i in out.issues]
    assert sev == sorted(sev, key={"error": 0, "warning": 1, "info": 2}.get)


# ---- date rules -------------------------------------------------------------------------------------
def test_expired_policy():
    out = engine.validate(ctx_for("insurance_policy", {**GOOD_POLICY, "policy_end_date": "2026-05-31"}))
    assert out.status == "needs_review" and "POLICY_EXPIRED" in codes(out)
    assert "31 May 2026" in next(i for i in out.issues if i.code == "POLICY_EXPIRED").message
    out = engine.validate(ctx_for("insurance_policy", {**GOOD_POLICY, "policy_end_date": "2026-06-01"}))
    assert "POLICY_EXPIRED" not in codes(out)  # expiring today is still in force


@pytest.mark.parametrize("bad", ["31/02/2025", "yesterday", "2025-13-01", "1850-01-01"])
def test_invalid_date_formats(bad):
    out = engine.validate(ctx_for("insurance_policy", {**GOOD_POLICY, "policy_start_date": bad}))
    assert "INVALID_DATE" in codes(out) and out.status == "needs_review"


def test_end_before_start_is_flagged():
    out = engine.validate(ctx_for("insurance_policy", {**GOOD_POLICY, "policy_start_date": "2027-04-01", "policy_end_date": "2027-03-01"}))
    assert "DATE_ORDER" in codes(out) and out.status == "needs_review"


def test_future_incident_and_invoice_dates():
    out = engine.validate(ctx_for("claim_form", {"claim_number": "C-1", "policy_number": "P-1", "customer_name": "A B",
                                                  "incident_date": "2026-07-01", "claim_amount": "500.00"}))
    assert "FUTURE_DATE" in codes(out)
    out = engine.validate(ctx_for("medical_bill", {"hospital_name": "H", "patient_name": "A B", "invoice_number": "1",
                                                   "invoice_date": "2030-01-01", "total_amount": "10.00"}))
    assert "FUTURE_DATE" in codes(out)


def test_identity_dates():
    base = {"document_type": "Driving Licence", "name": "A B", "id_number": "X1234"}
    assert "ID_EXPIRED" in codes(engine.validate(ctx_for("identity_document", {**base, "expiry_date": "2020-01-01"})))
    assert "INVALID_DATE" in codes(engine.validate(ctx_for("identity_document", {**base, "date_of_birth": "2030-01-01"})))
    assert "INVALID_DATE" in codes(engine.validate(ctx_for("identity_document", {**base, "date_of_birth": "1900-01-01"})))  # age > 110
    assert engine.validate(ctx_for("identity_document", {**base, "date_of_birth": "1990-05-05"})).status == "valid"


# ---- amounts, confidence, document level ------------------------------------------------------------------------
@pytest.mark.parametrize("bad,why", [("0.00", "greater than zero"), ("-50.00", "greater than zero"), ("abc", "not a valid amount"), ("9999999999.00", "unusually large")])
def test_invalid_amounts(bad, why):
    out = engine.validate(ctx_for("insurance_policy", {**GOOD_POLICY, "premium_amount": bad}))
    issue = next(i for i in out.issues if i.code == "INVALID_AMOUNT")
    assert why in issue.message and issue.field == "premium_amount"


def test_low_confidence_fields_unless_edited():
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, conf=0.45))
    assert codes(out).count("LOW_CONFIDENCE_FIELD") == 7 and out.status == "needs_review"
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, conf=0.45, edited=set(GOOD_POLICY)))
    assert out.status == "valid"  # a person has confirmed every value


def test_low_ocr_and_classification_confidence():
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, ocr_confidence=41.0))
    assert "LOW_OCR_CONFIDENCE" in codes(out) and out.status == "needs_review"
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, class_confidence=0.3))
    assert "LOW_CLASSIFICATION_CONFIDENCE" in codes(out)
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, class_confidence=0.3, class_method="manual"))
    assert out.status == "valid"  # a manual choice needs no confidence
    assert "NOT_CLASSIFIED" in codes(engine.validate(ctx_for(None, {})))


def test_no_text_found_and_other_category():
    out = engine.validate(ctx_for("other", {}, has_text=False, ocr_confidence=None))
    assert "NO_TEXT_FOUND" in codes(out) and out.status == "needs_review"
    assert engine.validate(ctx_for("other", {})).status == "valid"  # nothing required for supporting documents


def test_duplicates_and_policy_crosscheck():
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, exact_duplicates=[3]))
    assert "DUPLICATE_DOCUMENT" in codes(out) and "#3" in out.issues[0].message and out.status == "needs_review"
    out = engine.validate(ctx_for("insurance_policy", GOOD_POLICY, near_duplicates=[2]))
    assert "NEAR_DUPLICATE_DOCUMENT" in codes(out)
    claim = {"claim_number": "C-1", "policy_number": "POL-1", "customer_name": "A B", "incident_date": "2026-01-01", "claim_amount": "5.00"}
    matched = engine.validate(ctx_for("claim_form", claim, policy_match=7))
    assert "POLICY_MATCHED" in codes(matched) and matched.status == "valid"  # info never blocks
    assert "POLICY_NOT_FOUND" in codes(engine.validate(ctx_for("claim_form", claim)))


# ---- perceptual hash ----------------------------------------------------------------------------------------------------
def test_dhash_survives_noise_but_separates_layouts():
    a = make_page()
    ha = dhash(a)
    assert len(ha) == 16 and hamming(ha, ha) == 0
    assert hamming(ha, dhash(add_gaussian_noise(a, 12))) <= 4
    assert hamming(ha, dhash(make_page(lines=["CLAIM FORM", "Claim No: CLM-1", "Claimant: X Y", "Amount: 5000"] * 3))) > 15


# ---- API ------------------------------------------------------------------------------------------------------------------
def png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, lines, noise=0):
    page = make_page(lines=lines)
    if noise:
        page = add_gaussian_noise(page, noise)
    r = client.post(f"{P}/documents", files=[("files", ("doc.png", png(page), "image/png"))])
    return r.json()["items"][0]["id"]


@pytest.fixture()
def full_pipeline(monkeypatch):
    for name in ("auto_ocr", "auto_classify", "auto_extract", "auto_validate"):
        monkeypatch.setattr(settings, name, True)


POLICY_LINES = ["NORTHSTAR GENERAL INSURANCE", "POLICY SCHEDULE", "Policy No: POL-2025-001234", "Policyholder: Aarav Sharma",
                "Period of Insurance: 01/04/2025 to 31/03/2035", "Sum Insured: Rs. 500000", "Premium: Rs. 12450"]


def issues_of(client, doc_id):
    return {i["rule_code"]: i for i in client.get(f"{P}/documents/{doc_id}").json()["issues"]}


def test_upload_is_validated_and_marked_processed(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["status"] == "processed" and d["review_status"] in ("valid", "needs_review")
    assert client.get(f"{P}/documents/{doc_id}/status").json() == {"status": "processed", "stage": "done", "progress": 100}
    assert not any(i["rule_code"] in ("MISSING_FIELD", "POLICY_NUMBER_MISSING") for i in d["issues"])


def test_incomplete_document_lists_missing_fields(client, full_pipeline):
    doc_id = upload(client, ["POLICY SCHEDULE", "Policyholder: Aarav Sharma", "Premium: Rs. 12450"])
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["review_status"] == "incomplete"
    missing = {i["field_name"] for i in d["issues"] if i["severity"] == "error"}
    assert "policy_number" in missing and "policy_end_date" in missing
    assert "POLICY_NUMBER_MISSING" in {i["rule_code"] for i in d["issues"]}


def test_manual_correction_revalidates_and_can_complete_a_document(client, full_pipeline):
    doc_id = upload(client, ["POLICY SCHEDULE", "Policyholder: Aarav Sharma", "Premium: Rs. 12450", "Sum Insured: Rs. 500000"])
    assert client.get(f"{P}/documents/{doc_id}").json()["review_status"] == "incomplete"
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {
        "policy_number": "POL-2025-0001", "insurance_company": "Northstar General", "policy_start_date": "2025-04-01", "policy_end_date": "2035-03-31",
    }}).json()
    assert d["review_status"] == "valid", [i["rule_code"] for i in d["issues"]]  # human-confirmed values clear low-confidence warnings
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"policy_end_date": "2020-01-01"}}).json()
    assert d["review_status"] == "needs_review" and "POLICY_EXPIRED" in {i["rule_code"] for i in d["issues"]}
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"premium_amount": "-5"}}).json()
    assert "INVALID_AMOUNT" in {i["rule_code"] for i in d["issues"]}
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"policy_number": ""}}).json()
    assert d["review_status"] == "incomplete"  # clearing a mandatory field


def test_confirming_a_low_confidence_value_clears_its_warning(client, full_pipeline, monkeypatch):
    """The reviewer sees 'low confidence', checks the document, finds the value is right and confirms it."""
    from app.validation import rules

    monkeypatch.setattr(rules.Thresholds, "min_field_confidence", 0.99)  # force every untouched value to look uncertain
    monkeypatch.setattr(settings, "validation_min_field_confidence", 0.99)
    doc_id = upload(client, POLICY_LINES)
    d = client.get(f"{P}/documents/{doc_id}").json()
    low = [i for i in d["issues"] if i["rule_code"] == "LOW_CONFIDENCE_FIELD" and i["field_name"] == "policy_number"]
    assert low
    value = next(f["field_value"] for f in d["fields"] if f["field_name"] == "policy_number")
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"policy_number": value}}).json()  # same value = confirm
    assert not [i for i in d["issues"] if i["rule_code"] == "LOW_CONFIDENCE_FIELD" and i["field_name"] == "policy_number"]


def test_exact_duplicate_upload_is_flagged(client, full_pipeline):
    first = upload(client, POLICY_LINES)
    second = upload(client, POLICY_LINES)  # byte-identical image
    assert "DUPLICATE_DOCUMENT" not in issues_of(client, first)
    dup = issues_of(client, second)["DUPLICATE_DOCUMENT"]
    assert f"#{first}" in dup["message"] and dup["severity"] == "warning"


def test_rescan_is_flagged_but_similar_template_is_not(client, full_pipeline):
    first = upload(client, POLICY_LINES)
    rescan = upload(client, POLICY_LINES, noise=10)  # different bytes, same content
    other_values = [l.replace("Aarav Sharma", "Priya Nair").replace("POL-2025-001234", "POL-2024-778811") for l in POLICY_LINES]
    different = upload(client, other_values)  # same layout, different person and number
    assert "NEAR_DUPLICATE_DOCUMENT" in issues_of(client, rescan)
    assert not {"DUPLICATE_DOCUMENT", "NEAR_DUPLICATE_DOCUMENT"} & set(issues_of(client, different))
    assert first != rescan


def test_claim_matches_uploaded_policy(client, full_pipeline):
    upload(client, POLICY_LINES)
    claim = upload(client, ["CLAIM FORM", "Claim No: CLM-2025-00042", "Policy Number: POL-2025-001234", "Claimant Name: Aarav Sharma",
                            "Date of Incident: 12/05/2025", "Claim Amount: Rs. 80000"])
    assert "POLICY_MATCHED" in issues_of(client, claim)
    other = upload(client, ["CLAIM FORM", "Claim No: CLM-2025-00043", "Policy Number: POL-9999-000000", "Claimant Name: Aarav Sharma",
                            "Date of Incident: 12/05/2025", "Claim Amount: Rs. 80000"])
    assert "POLICY_NOT_FOUND" in issues_of(client, other)


def test_validate_endpoint(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    r = client.post(f"{P}/documents/{doc_id}/validate")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("valid", "needs_review") and set(body["counts"]) == {"error", "warning", "info"}
    assert "not a claim decision" in body["disclaimer"]
    assert client.post(f"{P}/documents/9999/validate").status_code == 404


def test_validate_with_nothing_to_validate(client):
    doc_id = upload(client, ["x"])  # OCR off, no category
    r = client.post(f"{P}/documents/{doc_id}/validate")
    assert r.status_code == 409 and r.json()["error"]["code"] == "NOTHING_TO_VALIDATE"


def test_reclassification_revalidates(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"category": "claim_form"}).json()
    assert d["review_status"] == "incomplete"  # claim fields are missing in a policy text
    assert "CLAIM_NUMBER_MISSING" in {i["rule_code"] for i in d["issues"]}


def test_stats_count_review_statuses(client, full_pipeline):
    upload(client, POLICY_LINES)
    upload(client, ["POLICY SCHEDULE", "Premium: Rs. 100"])
    s = client.get(f"{P}/stats/summary").json()
    assert s["incomplete"] >= 1 and s["valid"] + s["needs_review"] + s["incomplete"] == s["total"] == 2
