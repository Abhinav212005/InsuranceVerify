import cv2
import pytest

from app.classification.synthetic import generate
from app.core.config import settings
from app.extraction import engine
from app.extraction.extractors import extract
from app.extraction.normalizers import clean_name, find_amounts, find_dates, find_id, parse_amount
from app.extraction.schema import FIELDS, schema_payload
from tests.dip_helpers import make_page
from tests.handwritten_docs import DOCS
from tests.handwritten_truth import TRUTH

P = "/api/v1"


def fields_of(category, text, words=None, conf=95.0):
    pages = [{"page": 1, "text": text, "words": words or [], "avg_confidence": conf}]
    return {f.name: f for f in extract(category, pages)}


# ---- normalizers -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,iso",
    [
        ("12/05/2025", "2025-05-12"), ("12-05-2025", "2025-05-12"), ("2025-05-12", "2025-05-12"),
        ("12.05.2025", "2025-05-12"), ("1/2/25", "2025-02-01"), ("12 May 2025", "2025-05-12"),
        ("12th May, 2025", "2025-05-12"), ("May 12, 2025", "2025-05-12"), ("17 MAR 1988", "1988-03-17"),
        ("05/25/2025", "2025-05-25"),  # month-first is recognised when the second number is > 12
    ],
)
def test_date_formats(text, iso):
    assert find_dates(f"Date: {text} end")[0].iso == iso


def test_date_ocr_repair_and_invalid():
    h = find_dates("expires 3l/O3/2O26")[0]
    assert h.iso == "2026-03-31" and h.corrected
    bad = find_dates("on 31/02/2025")[0]
    assert bad.iso is None and bad.raw == "31/02/2025"  # looks like a date, is impossible
    assert find_dates("no dates here, only 12345") == []


@pytest.mark.parametrize(
    "text,value",
    [("Rs. 12,450.00", "12450.00"), ("Rs 5,00,000", "500000.00"), ("INR 1,00,00,000", "10000000.00"), ("₹ 750", "750.00"),
     ("$99.5", "99.50"), ("total 8,250", "8250.00"), ("Rs.12450", "12450.00")],
)
def test_amount_formats(text, value):
    assert parse_amount(text).value == value


def test_amount_edge_cases():
    assert find_amounts("GST 18% and 5% service") == []  # percentages are not money
    assert find_amounts("Table 5") == []  # lone small number without a currency
    assert find_amounts("date 12/05/2025") == []  # parts of dates are not amounts
    assert parse_amount("Rs. 1O,45O.00").value == "10450.00"  # OCR O -> 0
    assert parse_amount("paid 500 via card, Rs 900").value == "900.00"  # currency marker wins


def test_names_and_ids():
    assert clean_name("Mr Sanjay Kulkarni, Baner Road") == "Sanjay Kulkarni"
    assert clean_name("AARAV SHARMA") == "Aarav Sharma"
    assert clean_name("Rs 5000") is None and clean_name("12345") is None and clean_name("") is None
    assert find_id("POL-2025-001234 extra") == "POL-2025-001234"
    assert find_id("12/05/2025") is None  # a date is not an id
    assert find_id("Invoice total 7256", first_only=True) is None
    assert find_id("no digits here") is None


def test_real_world_ocr_damage_is_handled():
    """Cases seen when running Tesseract on rendered documents."""
    assert parse_amount("Premium: Rs. 31.600").value == "31600.00"  # thousands comma read as a period
    assert parse_amount("Rs. 4.50").value == "4.50"  # a genuine decimal is untouched
    bill = "Apex Heart Institute\nPatient Name: Divya Sharma\nUHID 810865\nTota! Amount Rs. 342.850"
    f = fields_of("medical_bill", bill)
    assert f["total_amount"].value == "342850.00"  # label typo tolerated
    assert fields_of("medical_bill", "Patient Name: A Rao\nUHID 810865\nRoom 12")["total_amount"].value is None  # an id is not money
    short = fields_of("repair_invoice", "Service Centre Bill\nInvoice total 7,000")["invoice_number"]
    assert short.value is None  # "invoice total" must not be fuzzy-matched as "invoice no"


def test_discharge_date_fallback_is_valid_and_explained():
    f = fields_of("medical_bill", "City Hospital\nPatient Name: A Rao\nDate of Admission 05-05-2023 Date of Discharge 2023-10-10\nTotal Amount Rs. 500")["invoice_date"]
    assert f.value == "2023-10-10" and f.note == "used the discharge date" and f.confidence > 0.5


def test_pan_repair_and_flagging():
    from app.extraction.extractors import repair_pan

    assert repair_pan("QIY0D7O17J") == "QIYOD7017J" and repair_pan("ABCDE1234F") == "ABCDE1234F" and repair_pan("123") is None
    text = "INCOME TAX DEPARTMENT\nPermanent Account Number\nQiyon70179\nName KARAN IYER"
    f = fields_of("identity_document", text)["id_number"]
    assert f.value == "QIYON70179" and "PAN format" in f.note and f.confidence < 0.5  # kept for the reviewer, flagged
    fixed = fields_of("identity_document", "Permanent Account Number\nABCD0 1234F\nName A B")["id_number"]
    assert fixed.value == "ABCDO1234F" or fixed.value.startswith("ABCD")


# ---- engine ---------------------------------------------------------------------------------
def test_label_flags_and_next_line_values():
    lines = engine.build_lines([{"page": 1, "text": "Insured Declared Value Rs 5\nName\nAarav Sharma\nSum Insured: 100000"}])
    assert engine.find_labeled(lines, ["^name!"]) == []  # "Name" alone has no colon after it
    hit = engine.find_labeled(lines, ["^name"])[0]
    assert hit.rest == "Aarav Sharma" and hit.next_line  # value on the following line
    assert engine.find_labeled(lines, ["sum insured!"])[0].rest == "100000"


def test_stop_labels_cut_multi_field_lines():
    f = fields_of("insurance_policy", "Policy No: POL-2025-0042   Policyholder: Aarav Sharma   Premium: Rs 1000")
    assert f["policy_number"].value == "POL-2025-0042" and f["customer_name"].value == "Aarav Sharma"


def test_locate_returns_box_and_confidence():
    words = [
        {"text": "Policy", "conf": 99, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.02, "line": 0, "block": 1},
        {"text": "No:", "conf": 99, "x": 0.21, "y": 0.1, "w": 0.05, "h": 0.02, "line": 0, "block": 1},
        {"text": "POL-1234", "conf": 40, "x": 0.3, "y": 0.1, "w": 0.2, "h": 0.02, "line": 0, "block": 1},
    ]
    bbox, conf = engine.locate(words, "POL-1234")
    assert bbox == [0.3, 0.1, 0.2, 0.02] and conf == 40
    assert engine.locate(words, "missing") == (None, None)


def test_confidence_reflects_ocr_quality_and_method():
    text = "Policy No: POL-2025-0042"
    good = fields_of("insurance_policy", text, conf=98)["policy_number"].confidence
    poor = fields_of("insurance_policy", text, conf=50)["policy_number"].confidence
    assert good > poor > 0
    word = {"text": "POL-2025-0042", "conf": 35, "x": 0.4, "y": 0.1, "w": 0.2, "h": 0.02, "line": 0, "block": 1}
    f = fields_of("insurance_policy", text, words=[word], conf=98)["policy_number"]
    assert f.bbox == [0.4, 0.1, 0.2, 0.02] and f.confidence < good  # a badly read word lowers confidence


# ---- extractors ---------------------------------------------------------------------------------
POLICY = """NORTHSTAR GENERAL INSURANCE
POLICY SCHEDULE
Policy No: POL-2025-001234
Policyholder: Aarav Sharma
Period of Insurance: 01/04/2025 to 3l/03/2026
Sum Insured: Rs. 5,00,000
Premium: Rs. 12,450.00"""


def test_policy_extraction_with_ocr_repair():
    f = fields_of("insurance_policy", POLICY)
    assert f["policy_number"].value == "POL-2025-001234" and f["customer_name"].value == "Aarav Sharma"
    assert f["insurance_company"].value == "Northstar General Insurance".upper().title() or f["insurance_company"].value
    assert (f["policy_start_date"].value, f["policy_end_date"].value) == ("2025-04-01", "2026-03-31")
    assert f["coverage_amount"].value == "500000.00" and f["premium_amount"].value == "12450.00"
    assert "corrected" in (f["policy_end_date"].note or "")
    assert f["policy_end_date"].confidence < f["policy_start_date"].confidence


def test_missing_fields_are_none_not_guessed():
    f = fields_of("insurance_policy", "Policy Schedule\nPremium: Rs 4000")
    assert f["premium_amount"].value == "4000.00"
    for name in ("policy_number", "customer_name", "policy_start_date", "policy_end_date", "coverage_amount"):
        assert f[name].value is None and f[name].confidence == 0.0
    assert set(f) == {s.name for s in FIELDS["insurance_policy"]}  # every expected field is listed


def test_invalid_date_is_kept_but_flagged():
    f = fields_of("claim_form", "CLAIM FORM\nClaim No: CLM-1234\nDate of Incident: 31/02/2025")["incident_date"]
    assert f.value == "31/02/2025" and "not a valid date" in f.note and f.confidence < 0.5


def test_claim_description_multiline_and_boilerplate():
    text = "CLAIM FORM\nDescription of Incident: the car was hit while\nparked outside the office\nI hereby declare the above is true"
    assert fields_of("claim_form", text)["claim_description"].value == "the car was hit while parked outside the office"


def test_identity_aadhaar_ignores_vid_and_finds_name():
    text = "Government of India\nRohit Pawar\nDOB: 21/11/1993\nMale\nVID: 9143 2288 7716 0042\n9143 2288 7716"
    f = fields_of("identity_document", text)
    assert f["document_type"].value == "Aadhaar Card" and f["id_number"].value == "9143 2288 7716"
    assert f["name"].value == "Rohit Pawar" and f["date_of_birth"].value == "1993-11-21"


def test_repair_vehicle_keeps_model_case_and_plate():
    f = fields_of("repair_invoice", "Ravi Motors\nTAX INVOICE\nVehicle Number MH12AB1234\nMake and Model Hyundai i20\nTotal 5,000")
    assert f["vehicle_details"].value == "Hyundai i20, MH12AB1234" and f["total_repair_cost"].value == "5000.00"


def test_other_category_has_no_fields():
    assert extract("other", [{"page": 1, "text": "dear sir", "words": [], "avg_confidence": 90}]) == []
    assert extract(None, [{"page": 1, "text": "x", "words": [], "avg_confidence": 90}]) == []


def test_multipage_values_report_their_page():
    pages = [
        {"page": 1, "text": "Policy Schedule\nPolicyholder: Aarav Sharma", "words": [], "avg_confidence": 95},
        {"page": 2, "text": "Annual Premium: Rs 9,000", "words": [], "avg_confidence": 95},
    ]
    f = {x.name: x for x in extract("insurance_policy", pages)}
    assert f["customer_name"].page == 1 and f["premium_amount"].page == 2


def test_schema_payload_matches_extractors():
    payload = schema_payload()
    assert set(payload) == set(FIELDS) and payload["other"] == []
    assert all({"name", "label", "kind", "mandatory"} == set(f) for fs in payload.values() for f in fs)


# ---- accuracy floors (guard against regressions; see scripts/evaluation/eval_extraction.py) ----------
def _accuracy(samples):
    import re

    norm = lambda v: None if v is None else re.sub(r"[^a-z0-9]", "", str(v).lower())
    ok = total = 0
    for label, text, truth in samples:
        got = {f.name: f.value for f in extract(label, [{"page": 1, "text": text, "words": [], "avg_confidence": 95.0}])}
        for k, v in truth.items():
            total += 1
            ok += norm(got.get(k)) == norm(v)
    return ok / total


def test_accuracy_on_synthetic_corpus_clean_and_noisy():
    clean = [(s.label, s.text, s.truth) for s in generate(8, noise=0.0, seed=21) if s.truth]
    noisy = [(s.label, s.text, s.truth) for s in generate(8, noise=0.04, seed=21) if s.truth]
    assert _accuracy(clean) >= 0.98
    assert _accuracy(noisy) >= 0.7  # strict exact-match under up to 4% character errors


def test_accuracy_on_handwritten_documents():
    docs = [(label, text, TRUTH[i]) for i, (label, text) in enumerate(DOCS) if TRUTH[i]]
    assert _accuracy(docs) >= 0.93


# ---- API ---------------------------------------------------------------------------------------------
def png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, lines):
    r = client.post(f"{P}/documents", files=[("files", ("doc.png", png(make_page(lines=lines)), "image/png"))])
    return r.json()["items"][0]["id"]


POLICY_LINES = ["NORTHSTAR GENERAL INSURANCE", "POLICY SCHEDULE", "Policy No: POL-2025-001234", "Policyholder: Aarav Sharma",
                "Period of Insurance: 01/04/2025 to 31/03/2026", "Sum Insured: Rs. 500000", "Premium: Rs. 12450"]


@pytest.fixture()
def full_pipeline(monkeypatch):
    for name in ("auto_ocr", "auto_classify", "auto_extract"):
        monkeypatch.setattr(settings, name, True)


def test_upload_extracts_fields_automatically(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["doc_category"] == "insurance_policy"
    by = {f["field_name"]: f for f in d["fields"]}
    assert [f["field_name"] for f in d["fields"]] == [s.name for s in FIELDS["insurance_policy"]]
    assert by["policy_number"]["field_value"] == "POL-2025-001234" and by["customer_name"]["field_value"] == "Aarav Sharma"
    assert by["policy_start_date"]["field_value"] == "2025-04-01" and by["premium_amount"]["field_value"] == "12450.00"
    assert by["policy_number"]["confidence"] > 0.6 and by["policy_number"]["is_edited"] is False
    meta = by["policy_number"]["bbox_json"]
    assert meta["method"] == "label" and meta["bbox"] and all(0 <= v <= 1 for v in meta["bbox"])
    assert client.get(f"{P}/documents/{doc_id}/status").json()["stage"] == "done"


def test_manual_edits_survive_reextraction(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"customer_name": "Aarav K. Sharma"}})
    r = client.post(f"{P}/documents/{doc_id}/extract", json={}).json()
    by = {f["field_name"]: f for f in r["fields"]}
    assert by["customer_name"]["field_value"] == "Aarav K. Sharma" and by["customer_name"]["is_edited"]
    assert by["policy_number"]["field_value"] == "POL-2025-001234"


def test_changing_category_reextracts_with_new_schema(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"customer_name": "Edited Name"}})
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"category": "claim_form"}).json()
    names = [f["field_name"] for f in d["fields"]]
    assert names == [s.name for s in FIELDS["claim_form"]]  # policy-only fields are gone
    by = {f["field_name"]: f for f in d["fields"]}
    assert by["customer_name"]["field_value"] == "Edited Name"  # shared field, edited value kept
    assert by["policy_number"]["field_value"] == "POL-2025-001234"
    d = client.patch(f"{P}/documents/{doc_id}/fields", json={"category": "other"}).json()
    assert [f["field_name"] for f in d["fields"]] == []  # "other" has no fields (unedited rows dropped)


def test_extract_endpoint_validation(client, full_pipeline):
    doc_id = upload(client, POLICY_LINES)
    assert client.post(f"{P}/documents/{doc_id}/extract", json={"category": "magic"}).status_code == 422
    assert client.post(f"{P}/documents/9999/extract", json={}).status_code == 404
    r = client.post(f"{P}/documents/{doc_id}/extract", json={"category": "claim_form"})
    assert r.status_code == 200 and r.json()["category"] == "claim_form"


def test_extract_without_text_is_refused(client):
    doc_id = upload(client, ["x"])  # OCR is off in tests
    r = client.post(f"{P}/documents/{doc_id}/extract", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "NO_TEXT"


def test_extraction_schema_endpoint(client):
    r = client.get(f"{P}/extraction/schema").json()
    assert [f["name"] for f in r["claim_form"]][:2] == ["claim_number", "policy_number"]
    assert r["insurance_policy"][-1] == {"name": "coverage_amount", "label": "Coverage amount", "kind": "amount", "mandatory": False}
