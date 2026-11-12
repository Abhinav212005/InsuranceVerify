"""Runs the OpenCV pipeline for stored document pages and records the results."""
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import AppError, NotFoundError
from app.db.models import Document, ProcessingResult
from app.dip import pipeline


def _page_root(doc_id: int, page_number: int) -> Path:
    return settings.storage_dir / "processed" / str(doc_id) / f"page_{page_number}"


def _load_original(doc_id: int, page_number: int) -> np.ndarray:
    path = _page_root(doc_id, page_number) / "original.png"
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise AppError("IMAGE_MISSING", f"Original image for page {page_number} is missing", 404)
    return img


def preprocess_page(
    doc: Document, row: ProcessingResult, recipe: str, params: dict[str, dict[str, Any]] | None, extras: bool = True
) -> dict[str, Any]:
    try:
        pipeline.resolve_params(pipeline.RECIPES[recipe], params)  # validate before doing heavy work
    except KeyError:
        raise AppError("UNKNOWN_RECIPE", f"Unknown recipe '{recipe}'. Options: {list(pipeline.RECIPES)}", 422)
    except pipeline.ParamError as e:
        raise AppError("INVALID_PARAMS", str(e), 422)

    n = row.page_number
    root = _page_root(doc.id, n)
    base = f"/files/{doc.id}/page_{n}"
    result = pipeline.run_pipeline(
        _load_original(doc.id, n), recipe, params, root / recipe, f"{base}/{recipe}", f"{base}/original.png", extras
    )
    # Summary stored in the database for the *active* recipe of this page.
    paths = {"original": f"{base}/original.png", "thumb": f"{base}/thumb.jpg"}
    paths.update({s["key"]: s["url"] for s in result["stages"] if s["key"] != "original"})
    row.recipe = recipe
    row.params_json = result["params"]
    row.stage_paths_json = {**paths, "extras": result.get("extras", {})}
    row.quality_before_json = result["metrics"]["before"]
    row.quality_after_json = result["metrics"]["after"]
    row.skew_angle = result["metrics"]["before"].get("skew_deg")
    row.preprocess_ms = result["total_ms"]
    result.pop("final_path", None)
    return result


def preprocess_document(
    doc: Document, recipe: str, params: dict[str, dict[str, Any]] | None, pages: list[int] | None = None
) -> list[dict[str, Any]]:
    rows = [r for r in doc.pages if pages is None or r.page_number in pages]
    if not rows:
        raise NotFoundError("Page")
    return [{"page": r.page_number, **preprocess_page(doc, r, recipe, params)} for r in rows]


def read_stages(doc: Document, page_number: int, recipe: str | None) -> dict[str, Any]:
    row = next((r for r in doc.pages if r.page_number == page_number), None)
    if row is None:
        raise NotFoundError("Page")
    recipe = recipe or row.recipe
    if not recipe or recipe not in pipeline.RECIPES:
        raise AppError("NOT_PROCESSED", "This page has not been preprocessed yet", 409)
    meta = _page_root(doc.id, page_number) / recipe / "meta.json"
    if not meta.exists():
        raise AppError("NOT_PROCESSED", f"Recipe '{recipe}' has not been run for this page", 409)
    data = json.loads(meta.read_text(encoding="utf-8"))
    data.pop("final_path", None)
    return {"page": page_number, "active_recipe": row.recipe, **data}
