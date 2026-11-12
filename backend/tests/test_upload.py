import io

import fitz
from PIL import Image

from app.core.config import settings

P = "/api/v1"


def png_bytes(size=(400, 300), color="white") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


def jpg_bytes(orientation: int | None = None) -> bytes:
    im = Image.new("RGB", (300, 200), "gray")
    b = io.BytesIO()
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(b, "JPEG", exif=exif)
    else:
        im.save(b, "JPEG")
    return b.getvalue()


def text_pdf(pages=1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        pg = doc.new_page()
        pg.insert_text((72, 100), "Policy Number POL-2025-001234 issued to Aarav Sharma for the period 2025", fontsize=11)
    return doc.tobytes()


def scanned_pdf() -> bytes:
    doc = fitz.open()
    pg = doc.new_page()
    pg.insert_image(pg.rect, stream=png_bytes((800, 1100), "lightgray"))
    return doc.tobytes()


def post(client, *files):
    return client.post(f"{P}/documents", files=[("files", f) for f in files])


def detail(client, doc_id):
    # TestClient runs background tasks before returning, so ingestion is already finished.
    return client.get(f"{P}/documents/{doc_id}").json()


def test_upload_png(client, tmp_storage):
    r = post(client, ("scan.png", png_bytes(), "image/png"))
    assert r.status_code == 201
    item = r.json()["items"][0]
    d = detail(client, item["id"])
    assert d["status"] == "ingested" and d["page_count"] == 1
    assert d["pages"][0]["stage_paths_json"]["original"] == f"/files/{item['id']}/page_1/original.png"
    assert (tmp_storage / "processed" / str(item["id"]) / "page_1" / "original.png").exists()
    assert (tmp_storage / "uploads" / str(item["id"]) / "original.png").exists()


def test_upload_jpg_exif_rotation_applied(client):
    r = post(client, ("phone.jpg", jpg_bytes(orientation=6), "image/jpeg"))  # 6 = rotate 90 deg
    doc_id = r.json()["items"][0]["id"]
    path = settings.storage_dir / "processed" / str(doc_id) / "page_1" / "original.png"
    with Image.open(path) as im:
        assert im.size == (200, 300)  # landscape 300x200 became portrait


def test_upload_multipage_digital_pdf_uses_text_layer(client, tmp_storage):
    r = post(client, ("policy.pdf", text_pdf(3), "application/pdf"))
    doc_id = r.json()["items"][0]["id"]
    d = detail(client, doc_id)
    assert d["status"] == "ingested" and d["page_count"] == 3 and len(d["pages"]) == 3
    assert [p["page_number"] for p in d["pages"]] == [1, 2, 3]
    assert all(p["ocr_source"] == "text_layer" and "POL-2025-001234" in p["ocr_text"] for p in d["pages"])
    for n in (1, 2, 3):
        assert (tmp_storage / "processed" / str(doc_id) / f"page_{n}" / "original.png").exists()


def test_upload_scanned_pdf_has_no_text_layer(client):
    doc_id = post(client, ("scan.pdf", scanned_pdf(), "application/pdf")).json()["items"][0]["id"]
    p = detail(client, doc_id)["pages"][0]
    assert p["ocr_source"] is None and p["ocr_text"] is None


def test_pdf_render_resolution_is_about_300dpi(client):
    doc_id = post(client, ("a4.pdf", text_pdf(), "application/pdf")).json()["items"][0]["id"]
    with Image.open(settings.storage_dir / "processed" / str(doc_id) / "page_1" / "original.png") as im:
        assert 2400 <= im.width <= 2600  # A4 (595 pt wide) at 300 DPI = ~2480 px


def test_rejects_wrong_extension(client):
    r = post(client, ("virus.exe", b"MZ\x90\x00" * 10, "application/octet-stream"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FILE"


def test_rejects_renamed_file(client):
    r = post(client, ("fake.pdf", png_bytes(), "application/pdf"))  # PNG content, .pdf name
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FILE"


def test_rejects_corrupt_files(client):
    r = post(client, ("bad.png", b"\x89PNG\r\n\x1a\n" + b"garbage", "image/png"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "CORRUPT_FILE"
    r = post(client, ("bad.pdf", b"%PDF-1.4 not really a pdf", "application/pdf"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "CORRUPT_FILE"


def test_rejects_empty_file(client):
    r = post(client, ("empty.png", b"", "image/png"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "EMPTY_FILE"


def test_rejects_oversize_file(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    r = post(client, ("big.png", png_bytes() + b"\0" * (2 * 1024 * 1024), "image/png"))
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_rejects_encrypted_pdf(client):
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    r = post(client, ("locked.pdf", data, "application/pdf"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "ENCRYPTED_PDF"


def test_rejects_too_many_pages(client):
    r = post(client, ("long.pdf", text_pdf(settings.max_pages + 1), "application/pdf"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "PAGE_LIMIT"


def test_mixed_batch_accepts_good_reports_bad(client):
    r = post(client, ("ok.png", png_bytes(), "image/png"), ("bad.txt", b"hello", "text/plain"))
    assert r.status_code == 201
    body = r.json()
    assert [i["filename"] for i in body["items"]] == ["ok.png"]
    assert body["rejected"][0]["filename"] == "bad.txt"


def test_duplicate_upload_is_flagged(client):
    data = png_bytes()
    first = post(client, ("a.png", data, "image/png")).json()["items"][0]
    second = post(client, ("copy.png", data, "image/png")).json()["items"][0]
    assert first["duplicate_of"] is None and second["duplicate_of"] == first["id"]


def test_filename_cannot_escape_storage(client, tmp_storage):
    r = post(client, ("../../evil.png", png_bytes(), "image/png"))
    assert r.json()["items"][0]["filename"] == "evil.png"
    assert not (tmp_storage.parent / "evil.png").exists()


def test_ingestion_failure_marks_document_failed(client, monkeypatch):
    from app.services import pdf_service

    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(pdf_service, "render_image", boom)
    doc_id = post(client, ("a.png", png_bytes(), "image/png")).json()["items"][0]["id"]
    d = detail(client, doc_id)
    assert d["status"] == "failed" and "boom" in d["error_message"]


def test_delete_removes_files(client, tmp_storage):
    doc_id = post(client, ("a.png", png_bytes(), "image/png")).json()["items"][0]["id"]
    assert (tmp_storage / "uploads" / str(doc_id)).exists()
    assert client.delete(f"{P}/documents/{doc_id}").status_code == 204
    assert not (tmp_storage / "uploads" / str(doc_id)).exists()
    assert not (tmp_storage / "processed" / str(doc_id)).exists()


def test_status_after_ingest(client):
    doc_id = post(client, ("a.png", png_bytes(), "image/png")).json()["items"][0]["id"]
    s = client.get(f"{P}/documents/{doc_id}/status").json()
    assert s == {"status": "ingested", "stage": "preprocessed", "progress": 33}
