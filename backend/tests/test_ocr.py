import io

import cv2
import fitz
import jiwer
import numpy as np
import pytest

from app.core.config import settings
from app.core.errors import AppError
from app.ocr import tesseract_engine as eng
from tests.dip_helpers import LINES, add_gaussian_noise, make_page, rotate_img, uneven_light

P = "/api/v1"
TRUTH = " ".join(t for i, t in enumerate(LINES * 2) if 120 + i * 70 <= 1520)


def norm(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def page():
    return make_page()


# ---- engine --------------------------------------------------------------------
def test_engine_reads_clean_page(page):
    r = eng.recognize(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), psm=6)
    assert jiwer.cer(TRUTH, norm(r["text"])) < 0.02
    assert r["avg_confidence"] > 80
    assert len(r["text"].splitlines()) >= 8  # lines preserved
    for w in r["words"]:
        assert w["text"] and 0 <= w["conf"] <= 100
        assert 0 <= w["x"] <= 1 and 0 <= w["y"] <= 1 and w["w"] > 0 and w["h"] > 0
        assert w["x"] + w["w"] <= 1.001 and w["y"] + w["h"] <= 1.001


def test_word_boxes_land_on_the_text(page):
    """A word's box (fractions) must cover dark pixels of the page at that position."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    r = eng.recognize(gray, psm=6)
    h, w = gray.shape
    w0 = next(x for x in r["words"] if x["text"].startswith("POL-2025"))
    crop = gray[int(w0["y"] * h): int((w0["y"] + w0["h"]) * h), int(w0["x"] * w): int((w0["x"] + w0["w"]) * w)]
    assert crop.size > 0 and (crop < 128).mean() > 0.05


def test_blank_page_returns_empty_result():
    r = eng.recognize(np.full((800, 600), 255, np.uint8))
    assert r == {"words": [], "text": "", "avg_confidence": 0.0}


def test_blur_lowers_confidence(page):
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    clean = eng.recognize(gray, 6)["avg_confidence"]
    blurry = eng.recognize(cv2.GaussianBlur(gray, (0, 0), 3.5), 6)
    assert blurry["avg_confidence"] < clean
    low_blurry = len(eng.low_confidence(blurry["words"], 60))
    low_clean = len(eng.low_confidence(eng.recognize(gray, 6)["words"], 60))
    assert low_blurry > low_clean


def test_invalid_psm_and_missing_tesseract(page, monkeypatch):
    with pytest.raises(AppError) as e:
        eng.recognize(page, psm=99)
    assert e.value.code == "INVALID_PSM"
    monkeypatch.setattr(settings, "tesseract_cmd", r"C:\nope\tesseract.exe")
    with pytest.raises(AppError) as e:
        eng.recognize(page, psm=3)
    assert e.value.code == "OCR_UNAVAILABLE"


def test_assemble_text_and_confidence_helpers():
    words = [
        {"text": "Hello", "conf": 90, "line": 0, "block": 1},
        {"text": "world", "conf": 50, "line": 0, "block": 1},
        {"text": "Next", "conf": 70, "line": 1, "block": 2},
    ]
    assert eng.assemble_text(words) == "Hello world\n\nNext"
    assert eng.average_confidence(words) == 70.0
    assert [w["text"] for w in eng.low_confidence(words, 60)] == ["world"]
    assert eng.average_confidence([]) == 0.0


# ---- API ---------------------------------------------------------------------------
def png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, name, data, mime="image/png"):
    r = client.post(f"{P}/documents", files=[("files", (name, data, mime))])
    assert r.status_code == 201
    return r.json()["items"][0]["id"]


@pytest.fixture()
def ocr_on(monkeypatch):
    monkeypatch.setattr(settings, "auto_ocr", True)


def test_upload_runs_ocr_automatically(client, ocr_on, page):
    doc_id = upload(client, "policy.png", png(rotate_img(page, 4)))
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["status"] == "processed"
    p = d["pages"][0]
    assert p["ocr_source"] == "ocr" and p["psm"] == settings.default_psm
    assert "POL-2025-001234" in p["ocr_text"] and p["avg_confidence"] > 70
    assert len(p["ocr_words_json"]) > 30
    assert client.get(f"{P}/documents/{doc_id}/status").json()["stage"] == "done"


def test_get_ocr_payload_and_threshold(client, ocr_on, page):
    doc_id = upload(client, "a.png", png(add_gaussian_noise(page, 20)))
    r = client.get(f"{P}/documents/{doc_id}/ocr").json()
    pg = r["pages"][0]
    assert r["threshold"] == 60 and pg["done"] and pg["source"] == "ocr"
    assert pg["image_url"].endswith(".png") and client.get(pg["image_url"]).status_code == 200
    assert pg["word_count"] == len(pg["words"]) > 0
    high = client.get(f"{P}/documents/{doc_id}/ocr", params={"threshold": 100}).json()
    assert high["low_conf_count"] == high["pages"][0]["word_count"]  # everything is below 100
    zero = client.get(f"{P}/documents/{doc_id}/ocr", params={"threshold": 0}).json()
    assert zero["low_conf_count"] == 0
    assert client.get(f"{P}/documents/{doc_id}/ocr", params={"threshold": 500}).status_code == 422


def test_ocr_not_run_when_disabled(client, page):
    doc_id = upload(client, "a.png", png(page))
    pg = client.get(f"{P}/documents/{doc_id}/ocr").json()["pages"][0]
    assert pg["done"] is False and pg["text"] == "" and pg["words"] == []


def test_manual_ocr_with_other_psm_and_source(client, page):
    doc_id = upload(client, "a.png", png(page))
    r = client.post(f"{P}/documents/{doc_id}/ocr", json={"psm": 6, "source": "original"})
    assert r.status_code == 200
    pg = r.json()["pages"][0]
    assert pg["psm"] == 6 and pg["image_url"].endswith("/original.png") and "Policy Number" in pg["text"]
    r = client.post(f"{P}/documents/{doc_id}/ocr", json={"psm": 11, "source": "processed"})
    assert r.json()["pages"][0]["image_url"].endswith("/final.png")


def test_ocr_request_validation(client, page):
    doc_id = upload(client, "a.png", png(page))
    r = client.post(f"{P}/documents/{doc_id}/ocr", json={"psm": 99})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_PSM"
    assert client.post(f"{P}/documents/{doc_id}/ocr", json={"source": "nonsense"}).status_code == 422
    assert client.post(f"{P}/documents/{doc_id}/ocr", json={"recipe": "magic"}).status_code == 422
    assert client.post(f"{P}/documents/{doc_id}/ocr", json={"pages": [9]}).status_code == 404
    assert client.post(f"{P}/documents/9999/ocr", json={}).status_code == 404
    assert client.get(f"{P}/documents/9999/ocr").status_code == 404


def test_ocr_refused_before_pages_exist(client, db):
    from tests.conftest import make_doc

    doc = make_doc(db, status="uploaded", review=None)
    r = client.post(f"{P}/documents/{doc.id}/ocr", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "NOT_READY"


def test_ocr_failure_is_not_fatal(client, ocr_on, page, monkeypatch):
    monkeypatch.setattr(settings, "tesseract_cmd", r"C:\nope\tesseract.exe")
    doc_id = upload(client, "a.png", png(page))
    d = client.get(f"{P}/documents/{doc_id}").json()
    assert d["status"] == "ingested" and "OCR failed" in d["error_message"]
    assert d["pages"][0]["stage_paths_json"]["gray"]  # preprocessing results survived
    assert client.get(f"{P}/documents/{doc_id}/status").json()["stage"] == "ocr_failed"


def test_digital_pdf_keeps_text_layer_and_gets_word_boxes(client, ocr_on):
    pdf = fitz.open()
    pg = pdf.new_page()
    pg.insert_text((72, 100), "Policy Number POL-2025-001234 issued to Aarav Sharma in the year 2025", fontsize=11)
    doc_id = upload(client, "digital.pdf", pdf.tobytes(), "application/pdf")
    p = client.get(f"{P}/documents/{doc_id}/ocr").json()["pages"][0]
    assert p["source"] == "text_layer" and p["psm"] is None and p["avg_confidence"] == 100.0
    assert "POL-2025-001234" in p["text"]
    w = next(x for x in p["words"] if x["text"] == "POL-2025-001234")
    assert w["conf"] == 100.0 and 0 < w["x"] < 0.5 and 0 < w["y"] < 0.2  # normalized to the page
    assert p["image_url"].endswith("/original.png")

    # OCR does not overwrite the text layer unless forced
    again = client.post(f"{P}/documents/{doc_id}/ocr", json={}).json()["pages"][0]
    assert again["source"] == "text_layer"
    forced = client.post(f"{P}/documents/{doc_id}/ocr", json={"force": True, "psm": 6}).json()["pages"][0]
    assert forced["source"] == "ocr" and forced["psm"] == 6 and "POL-2025-001234" in forced["text"]


def test_multipage_scanned_pdf_pagewise_text(client, ocr_on, page):
    pdf = fitz.open()
    for text in ("FIRST PAGE alpha", "SECOND PAGE bravo"):
        img = np.full((1600, 1200, 3), 255, np.uint8)
        cv2.putText(img, text, (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
        p = pdf.new_page()
        p.insert_image(p.rect, stream=png(img))
    doc_id = upload(client, "scan.pdf", pdf.tobytes(), "application/pdf")
    pages = client.get(f"{P}/documents/{doc_id}/ocr").json()["pages"]
    assert [p["page"] for p in pages] == [1, 2]
    assert "FIRST" in pages[0]["text"] and "SECOND" not in pages[0]["text"]
    assert "SECOND" in pages[1]["text"] and all(p["source"] == "ocr" for p in pages)


def test_preprocessing_fixes_a_page_ocr_cannot_read_raw(client, page):
    """The core DIP claim: a shadowed page is unreadable raw but fine after processing."""
    shadowed = uneven_light(rotate_img(page, -3), 0.7)
    doc_id = upload(client, "shadow.png", png(shadowed))

    def cer(source):
        r = client.post(f"{P}/documents/{doc_id}/ocr", json={"psm": 6, "source": source}).json()
        return jiwer.cer(TRUTH, norm(r["pages"][0]["text"]))

    raw, processed = cer("original"), cer("processed")
    assert raw > 0.3 and processed < 0.05, (raw, processed)
