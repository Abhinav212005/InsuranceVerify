"""Shared pieces for the evaluation experiments: dataset access, processing variants, OCR scoring,
and a small parallel runner."""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # backend/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))

import cv2
import jiwer
import numpy as np

import degrade
from app.core.config import settings
from app.core.errors import AppError
from app.dip import metrics
from app.dip.pipeline import RECIPES, Recipe, process_in_memory
from app.dip.steps import basic, geometry
from app.ocr import tesseract_engine

PROJECT = ROOT.parent
DATASET = PROJECT / "dataset" / "synthetic"
RESULTS = PROJECT / "evaluation_results"
CHARTS = RESULTS / "charts"

# ---- processing variants used by several experiments ------------------------------------------
_STD = ("gray", "denoise", "threshold", "deskew", "morphology", "resize")


def _register(name: str, steps: tuple[str, ...], overrides: dict | None = None) -> str:
    RECIPES[name] = Recipe(name, name, "evaluation variant", steps, overrides or {})
    return name


# cumulative ablation: each row adds one technique to the previous one
ABLATION = [
    ("A0 raw image", None),
    ("A1 + grayscale", _register("abl_gray", ("gray",))),
    ("A2 + median denoise", _register("abl_denoise", ("gray", "denoise"))),
    ("A3 + CLAHE", _register("abl_clahe", ("gray", "denoise", "contrast"))),
    ("A4 + adaptive threshold", _register("abl_thresh", ("gray", "denoise", "contrast", "threshold"))),
    ("A5 + deskew", _register("abl_deskew", ("gray", "denoise", "contrast", "threshold", "deskew"))),
    ("A6 + morphology", _register("abl_morph", ("gray", "denoise", "contrast", "threshold", "deskew", "morphology"))),
    ("A7 + resize (R4 standard+CLAHE)", _register("abl_resize", ("gray", "denoise", "contrast", "threshold", "deskew", "morphology", "resize"))),
    ("R3 default (no CLAHE)", "standard"),
    ("R5 full (+ page boundary)", "full"),
]
_register("thr_global", ("gray", "denoise", "threshold"), {"threshold": {"method": "global"}})
_register("thr_otsu", ("gray", "denoise", "threshold"), {"threshold": {"method": "otsu"}})
_register("thr_adaptive", ("gray", "denoise", "threshold"), {"threshold": {"method": "adaptive"}})
for _name, _den in {"den_none": {"method": "none"}, "den_gauss": {"method": "gaussian", "ksize": 5},
                    "den_median3": {"method": "median", "ksize": 3}, "den_median5": {"method": "median", "ksize": 5},
                    "den_nlm": {"method": "nlm", "h": 10.0}}.items():
    _register(_name, ("gray", "denoise", "threshold"), {"denoise": _den})
_register("no_deskew", ("gray", "denoise", "threshold", "morphology", "resize"))
_register("gray_only", ("gray",))
_register("gray_upscale", ("gray", "resize"), {"resize": {"min_width": 1654, "max_width": 4000}})


# ---- dataset -------------------------------------------------------------------------------------
def read_manifest() -> list[dict]:
    with open(DATASET / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["doc_index"], r["seed"] = int(r["doc_index"]), int(r["seed"])
        r["meta"] = json.loads(r["meta"])
    return rows


def pick(rows: list[dict], conditions: list[str], per_category: int) -> list[dict]:
    """First `per_category` documents of every category, restricted to `conditions`."""
    counts: dict[str, set] = {}
    out = []
    for r in rows:
        if r["condition"] not in conditions:
            continue
        seen = counts.setdefault(r["category"], set())
        if r["id"] not in seen and len(seen) >= per_category:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def truth_of(doc_id: str) -> dict:
    return json.loads((DATASET / "truth" / f"{doc_id}.json").read_text(encoding="utf-8"))


def load_image(row: dict, scale: float | None = None) -> np.ndarray:
    clean = cv2.imread(str(DATASET / "clean" / f"{row['id']}.png"), cv2.IMREAD_COLOR)
    img, _ = degrade.apply(row["condition"], clean, row["doc_index"], row["seed"])
    if scale and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


# ---- scoring ---------------------------------------------------------------------------------------
def norm_text(t: str) -> str:
    return " ".join(t.split())


def cer_wer(truth: str, hyp: str) -> tuple[float, float]:
    truth, hyp = norm_text(truth), norm_text(hyp)
    if not hyp:
        return 1.0, 1.0
    return float(jiwer.cer(truth, hyp)), float(jiwer.wer(truth, hyp))


def ocr(img: np.ndarray, psm: int = 3) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    r = tesseract_engine.recognize(img, psm)
    return r["text"], r["avg_confidence"], (time.perf_counter() - t0) * 1000


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two convex quadrilaterals (float32, 4x2)."""
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


# ---- one unit of work: process an image, OCR it, score it ----------------------------------------------
def run_task(spec: dict) -> dict:
    """spec: row fields + recipe (None = raw), params, psm, scale, extras flags. Runs in a worker process."""
    row = spec["row"]
    truth = truth_of(row["id"])
    img = load_image(row, spec.get("scale"))
    t0 = time.perf_counter()
    recipe = spec.get("recipe")
    out = {"id": row["id"], "category": row["category"], "template": row["template"], "condition": row["condition"],
           "variant": spec.get("variant", recipe or "raw"), "psm": spec.get("psm", 3), "scale": spec.get("scale") or 1.0}
    if recipe is None:
        final, gray_ref, info = img, basic.to_gray(img), {}
        out["dip_ms"] = 0.0
    else:
        p = process_in_memory(img, recipe, spec.get("params"))
        final, gray_ref, info = p.final, p.gray_ref, p.info
        out["dip_ms"] = (time.perf_counter() - t0) * 1000
        out["step_ms"] = {s["name"]: round(s["ms"], 1) for s in p.steps}
    try:
        text, conf, ocr_ms = ocr(final, spec.get("psm", 3))
        out["timeout"] = 0
    except AppError:  # Tesseract gave up (e.g. a page full of speckle): counts as a total failure, not a crash
        text, conf, ocr_ms = "", 0.0, settings.ocr_timeout_s * 1000.0
        out["timeout"] = 1
    out["cer"], out["wer"] = cer_wer(truth["text"], text)
    out["ocr_conf"], out["ocr_ms"] = conf, ocr_ms
    out["empty"] = int(not text.strip())

    if spec.get("quality"):  # image-fidelity metrics against the clean original (same geometry only)
        clean = cv2.imread(str(DATASET / "clean" / f"{row['id']}.png"), cv2.IMREAD_GRAYSCALE)
        if gray_ref.shape == clean.shape:
            small = lambda g: cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            out.update(metrics.psnr_ssim(small(clean), small(gray_ref)))
        out["sharpness"] = metrics.sharpness(gray_ref)
        out["noise"] = metrics.noise_sigma(gray_ref)
    if spec.get("skew"):
        true = row["meta"].get("angle", 0.0)
        est = -info.get("deskew_correction_deg", 0.0) if "deskew_correction_deg" in info else metrics.skew_degrees(basic.to_gray(img))
        out.update(true_angle=true, est_angle=est, angle_error=abs(est - true))
    if spec.get("boundary"):
        quad = geometry.find_document_quad(img)
        corners = np.array(row["meta"]["corners"], np.float32) if "corners" in row["meta"] else None
        out["found"] = int(quad is not None)
        out["iou"] = quad_iou(geometry.order_points(quad), geometry.order_points(corners)) if quad is not None and corners is not None else 0.0
    if spec.get("text"):
        out["text"] = text
    return out


def _init_worker() -> None:
    settings.ocr_timeout_s = 40  # evaluation only: a hung page must not stall the whole run
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per process: more processes, no oversubscription


def parallel(fn, tasks: list, workers: int = 10, label: str = "") -> list:
    start, results = time.time(), []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.map(fn, tasks, chunksize=2), 1):
            results.append(r)
            if i % 100 == 0 or i == len(tasks):
                print(f"  {label}: {i}/{len(tasks)} ({time.time() - start:.0f}s)", flush=True)
    return results


def write_csv(name: str, rows: list[dict], drop: tuple = ("step_ms", "text")) -> Path:
    RESULTS.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r if k not in drop})
    path = RESULTS / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def mean(vals) -> float:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")
