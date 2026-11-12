import json

import cv2
import fitz
import pytest

from app.core.config import settings
from tests.dip_helpers import make_page

P = "/api/v1"
LINES = ["NORTHSTAR GENERAL INSURANCE", "POLICY SCHEDULE", "Policy No: POL-2025-001234", "Policyholder: Aarav Sharma",
         "Period of Insurance: 01/04/2025 to 31/03/2035", "Sum Insured: Rs. 500000", "Premium: Rs. 12450"]


@pytest.fixture()
def full_pipeline(monkeypatch):
    for name in ("auto_ocr", "auto_classify", "auto_extract", "auto_validate"):
        monkeypatch.setattr(settings, name, True)


def upload(client, lines=LINES):
    ok, buf = cv2.imencode(".png", make_page(lines=lines))
    r = client.post(f"{P}/documents", files=[("files", ("policy.png", buf.tobytes(), "image/png"))])
    return r.json()["items"][0]["id"]


def test_json_report(client, full_pipeline):
    doc_id = upload(client)
    r = client.get(f"{P}/documents/{doc_id}/report")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    rep = json.loads(r.content)
    assert rep["document"]["filename"] == "policy.png" and rep["classification"]["category"] == "insurance_policy"
    assert rep["review_status"] in ("valid", "needs_review", "incomplete")
    by = {f["name"]: f for f in rep["fields"]}
    assert by["policy_number"]["label"] == "Policy number" and by["customer_name"]["value"] == "Aarav Sharma"
    assert "POL-2025-001234" in rep["ocr"][0]["text"]
    assert "not an approval" in rep["disclaimer"]


def test_pdf_report_is_a_readable_pdf(client, full_pipeline):
    doc_id = upload(client)
    r = client.get(f"{P}/documents/{doc_id}/report", params={"format": "pdf"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    with fitz.open(stream=r.content, filetype="pdf") as pdf:
        text = "".join(p.get_text() for p in pdf)
    for expected in ("Document report", "Aarav Sharma", "Policy number", "POL-2025-001234", "Recognised text", "not an approval"):
        assert expected in text


def test_pdf_report_spans_pages_for_long_text(client, db):
    from tests.conftest import make_doc

    doc = make_doc(db)
    doc.pages[0].ocr_text = "\n".join(f"line {i} of a long document" for i in range(400))
    db.commit()
    r = client.get(f"{P}/documents/{doc.id}/report", params={"format": "pdf"})
    with fitz.open(stream=r.content, filetype="pdf") as pdf:
        assert pdf.page_count >= 2


def test_report_errors(client):
    assert client.get(f"{P}/documents/9999/report").status_code == 404
    assert client.get(f"{P}/documents/1/report", params={"format": "docx"}).status_code == 422


def test_reprocess_runs_the_pipeline_again(client, full_pipeline):
    doc_id = upload(client)
    client.patch(f"{P}/documents/{doc_id}/fields", json={"fields": {"customer_name": "Edited Name"}})
    r = client.post(f"{P}/documents/{doc_id}/process")
    assert r.status_code == 202 and r.json() == {"id": doc_id, "status": "processing"}
    d = client.get(f"{P}/documents/{doc_id}").json()  # the background job has finished by now
    assert d["status"] == "processed" and d["review_status"] in ("valid", "needs_review")
    by = {f["field_name"]: f for f in d["fields"]}
    assert by["customer_name"]["field_value"] == "Edited Name"  # a person's correction survives
    assert client.post(f"{P}/documents/9999/process").status_code == 404


def test_reprocess_recovers_a_failed_document(client, monkeypatch):
    from app.services import pdf_service

    original = pdf_service.render_image
    monkeypatch.setattr(pdf_service, "render_image", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    doc_id = upload(client)
    assert client.get(f"{P}/documents/{doc_id}").json()["status"] == "failed"
    monkeypatch.setattr(pdf_service, "render_image", original)
    client.post(f"{P}/documents/{doc_id}/process")
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["status"] == "ingested" and d["error_message"] is None
