from sqlalchemy import select

from app.db.models import ExtractedInformation, ProcessingResult, ValidationResult
from tests.conftest import make_doc

P = "/api/v1"


def test_health(client):
    r = client.get(f"{P}/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_list_empty(client):
    r = client.get(f"{P}/documents")
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0, "page": 1, "size": 20}


def test_list_filters_and_pagination(client, db):
    make_doc(db, "policy_a.pdf", "insurance_policy", "valid")
    make_doc(db, "claim_b.png", "claim_form", "needs_review")
    make_doc(db, "bill_c.jpg", "medical_bill", "incomplete", days_ago=10)

    assert client.get(f"{P}/documents").json()["total"] == 3
    assert client.get(f"{P}/documents", params={"type": "claim_form"}).json()["total"] == 1
    assert client.get(f"{P}/documents", params={"status": "incomplete"}).json()["items"][0]["filename"] == "bill_c.jpg"
    assert client.get(f"{P}/documents", params={"q": "policy"}).json()["total"] == 1
    # search also matches extracted field values
    assert client.get(f"{P}/documents", params={"q": "POL-1"}).json()["total"] == 3
    r = client.get(f"{P}/documents", params={"size": 2, "page": 2}).json()
    assert r["total"] == 3 and len(r["items"]) == 1


def test_list_date_filter(client, db):
    make_doc(db, "new.pdf", days_ago=0)
    make_doc(db, "old.pdf", days_ago=30)
    from datetime import datetime, timedelta, timezone

    start = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
    r = client.get(f"{P}/documents", params={"from": start}).json()
    assert [d["filename"] for d in r["items"]] == ["new.pdf"]


def test_list_rejects_bad_params(client):
    r = client.get(f"{P}/documents", params={"type": "banana"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.get(f"{P}/documents", params={"size": 1000}).status_code == 422


def test_get_detail(client, db):
    doc = make_doc(db)
    r = client.get(f"{P}/documents/{doc.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "policy_1.pdf"
    assert body["pages"][0]["ocr_text"] == "hello"
    assert body["fields"][0]["field_name"] == "policy_number"
    assert body["issues"][0]["rule_code"] == "LOW_CONFIDENCE_FIELD"


def test_get_missing_returns_error_envelope(client):
    r = client.get(f"{P}/documents/999")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "NOT_FOUND", "message": "Document not found"}}


def test_status_endpoint(client, db):
    done = make_doc(db, "a.pdf")
    running = make_doc(db, "b.pdf", status="processing", current_stage="ocr", review=None)
    assert client.get(f"{P}/documents/{done.id}/status").json() == {
        "status": "processed", "stage": "done", "progress": 100,
    }
    s = client.get(f"{P}/documents/{running.id}/status").json()
    assert s["stage"] == "ocr" and s["progress"] == 42


def test_patch_fields_edits_and_creates(client, db):
    doc = make_doc(db)
    r = client.patch(
        f"{P}/documents/{doc.id}/fields",
        json={"fields": {"policy_number": " POL-999 ", "customer_name": "Aarav Sharma"}, "category": "claim_form"},
    )
    assert r.status_code == 200
    body = r.json()
    by = {f["field_name"]: f for f in body["fields"]}
    assert by["policy_number"]["field_value"] == "POL-999"
    assert by["policy_number"]["original_value"] == "POL-1"  # auto-extracted value is kept
    assert by["policy_number"]["is_edited"] is True
    assert by["customer_name"]["is_edited"] is True
    assert body["doc_category"] == "claim_form"


def test_patch_with_unchanged_value_counts_as_confirmation(client, db):
    doc = make_doc(db)
    f = client.patch(f"{P}/documents/{doc.id}/fields", json={"fields": {"policy_number": "POL-1"}}).json()["fields"][0]
    assert f["is_edited"] is True and f["confidence"] == 1.0 and f["field_value"] == f["original_value"] == "POL-1"


def test_patch_clearing_value_sets_null(client, db):
    doc = make_doc(db)
    r = client.patch(f"{P}/documents/{doc.id}/fields", json={"fields": {"policy_number": "  "}})
    assert r.json()["fields"][0]["field_value"] is None


def test_patch_invalid_category(client, db):
    doc = make_doc(db)
    assert client.patch(f"{P}/documents/{doc.id}/fields", json={"category": "nope"}).status_code == 422


def test_delete_cascades(client, db, session_factory):
    doc = make_doc(db)
    assert client.delete(f"{P}/documents/{doc.id}").status_code == 204
    assert client.get(f"{P}/documents/{doc.id}").status_code == 404
    with session_factory() as s:  # child rows are gone too
        for model in (ProcessingResult, ExtractedInformation, ValidationResult):
            assert s.scalars(select(model)).first() is None
    assert client.delete(f"{P}/documents/{doc.id}").status_code == 404


def test_stats_summary(client, db):
    make_doc(db, "a.pdf", "insurance_policy", "valid", processing_ms=2000)
    make_doc(db, "b.pdf", "insurance_policy", "needs_review", processing_ms=4000)
    make_doc(db, "c.pdf", "claim_form", "incomplete", processing_ms=6000)
    make_doc(db, "d.pdf", "claim_form", None, status="failed", processing_ms=None)
    s = client.get(f"{P}/stats/summary").json()
    assert (s["total"], s["valid"], s["needs_review"], s["incomplete"], s["failed"]) == (4, 1, 1, 1, 1)
    assert s["avg_processing_ms"] == 4000
    assert {d["type"]: d["count"] for d in s["distribution"]} == {"insurance_policy": 2, "claim_form": 2}
    assert len(s["per_day"]) == 7 and s["per_day"][-1]["uploads"] == 4


def test_cors_allows_frontend_origin(client):
    r = client.options(
        f"{P}/documents",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
