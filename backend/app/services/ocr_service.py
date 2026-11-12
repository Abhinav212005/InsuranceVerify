"""Runs OCR for stored pages and records text, word boxes and confidences."""
import time
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from app.core.config import settings
from app.core.errors import AppError, NotFoundError
from app.db.models import Document, ProcessingResult
from app.dip import pipeline as dip_pipeline
from app.ocr import tesseract_engine as engine
from app.services import dip_service

# Preferred image for OCR, most processed first (later stages exist only if the recipe ran them).
PROCESSED_PREFERENCE = ["final", "deskewed", "thresholded", "denoised", "gray"]


def _pick_image(doc: Document, row: ProcessingResult, source: str, recipe: str | None) -> tuple[Path, str]:
    """Returns (file path, url) of the image that will be OCR'd."""
    root = dip_service._page_root(doc.id, row.page_number)
    base = f"/files/{doc.id}/page_{row.page_number}"
    if source == "processed":
        recipe = recipe or row.recipe
        if recipe in dip_pipeline.RECIPES:
            for stage in PROCESSED_PREFERENCE:
                p = root / recipe / f"{stage}.png"
                if p.exists():
                    return p, f"{base}/{recipe}/{stage}.png"
    return root / "original.png", f"{base}/original.png"


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise AppError("IMAGE_MISSING", f"Image {path.name} is missing", 404)
    return img


def _text_layer_words(doc: Document, page_number: int) -> list[dict]:
    """Word boxes straight from the PDF's embedded text (no OCR needed, confidence 100)."""
    path = settings.storage_dir / doc.stored_path
    words: list[dict] = []
    with fitz.open(path) as pdf:
        page = pdf[page_number - 1]
        w, h = page.rect.width, page.rect.height
        lines: dict[tuple[int, int], int] = {}
        for x0, y0, x1, y1, text, block, line, _ in page.get_text("words"):
            text = text.strip()
            if not text:
                continue
            idx = lines.setdefault((block, line), len(lines))
            words.append(
                {
                    "text": text, "conf": 100.0, "line": idx, "block": block,
                    "x": round(x0 / w, 4), "y": round(y0 / h, 4),
                    "w": round((x1 - x0) / w, 4), "h": round((y1 - y0) / h, 4),
                }
            )
    return words


def ocr_page(
    doc: Document, row: ProcessingResult, psm: int | None = None, source: str = "processed",
    recipe: str | None = None, force: bool = False,
) -> None:
    psm = engine.check_psm(psm or settings.default_psm)
    started = time.perf_counter()
    paths = dict(row.stage_paths_json or {})

    if doc.file_type == "pdf" and row.ocr_source == "text_layer" and row.ocr_text and not force:
        # Digital PDF page: keep the embedded text, but compute word boxes for the overlay.
        words = _text_layer_words(doc, row.page_number)
        row.ocr_words_json = words
        row.avg_confidence = 100.0
        row.low_conf_count = 0
        row.psm = None
        paths["ocr_image"] = f"/files/{doc.id}/page_{row.page_number}/original.png"
    else:
        image_path, url = _pick_image(doc, row, source, recipe)
        result = engine.recognize(_read_gray(image_path), psm)
        row.ocr_text = result["text"]
        row.ocr_words_json = result["words"]
        row.avg_confidence = result["avg_confidence"]
        row.low_conf_count = len(engine.low_confidence(result["words"], settings.ocr_low_conf_threshold))
        row.psm = psm
        row.ocr_source = "ocr"
        paths["ocr_image"] = url
    row.stage_paths_json = paths  # reassign so SQLAlchemy sees the JSON change
    row.ocr_ms = int((time.perf_counter() - started) * 1000)


def ocr_document(
    doc: Document, psm: int | None, source: str, recipe: str | None, pages: list[int] | None, force: bool
) -> None:
    rows = [r for r in doc.pages if pages is None or r.page_number in pages]
    if not rows:
        raise NotFoundError("Page")
    for row in rows:
        ocr_page(doc, row, psm, source, recipe, force)


def page_payload(doc: Document, row: ProcessingResult, threshold: float) -> dict[str, Any]:
    words = row.ocr_words_json or []
    low = engine.low_confidence(words, threshold)
    return {
        "page": row.page_number,
        "done": row.ocr_source is not None,
        "source": row.ocr_source,  # "ocr" or "text_layer"
        "psm": row.psm,
        "image_url": (row.stage_paths_json or {}).get("ocr_image"),
        "avg_confidence": row.avg_confidence,
        "word_count": len(words),
        "low_conf_count": len(low),
        "text": row.ocr_text or "",
        "words": words,
        "ocr_ms": row.ocr_ms,
    }


def document_payload(doc: Document, threshold: float | None = None) -> dict[str, Any]:
    t = settings.ocr_low_conf_threshold if threshold is None else threshold
    pages = [page_payload(doc, r, t) for r in doc.pages]
    done = [p for p in pages if p["done"]]
    avg = round(sum(p["avg_confidence"] or 0 for p in done) / len(done), 1) if done else None
    return {
        "document_id": doc.id,
        "threshold": t,
        "avg_confidence": avg,
        "low_conf_count": sum(p["low_conf_count"] for p in pages),
        "psm_options": engine.ALLOWED_PSM,
        "pages": pages,
    }
