"""Modular preprocessing pipeline.

Every step is a small function on a shared context. A *recipe* is an ordered list of
steps (with optional parameter overrides), so different preprocessing strategies can be
run on the same page and compared. The runner saves one image per UI stage plus extra
visualizations, measures quality before/after, and writes everything to meta.json.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.dip import metrics
from app.dip.steps import basic, deskew, geometry, morphology, threshold

STAGE_ORDER = ["original", "gray", "denoised", "thresholded", "deskewed", "final"]
STAGE_LABEL = {
    "original": "Original", "gray": "Grayscale", "denoised": "Denoised",
    "thresholded": "Thresholded", "deskewed": "Deskewed", "final": "Final",
}
EXTRA_WIDTH = 1000  # extras are rendered at this width to keep disk usage small


@dataclass
class Ctx:
    bgr: np.ndarray
    img: np.ndarray | None = None
    last_gray: np.ndarray | None = None  # most recent non-binary image (for quality metrics)
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    binarized: bool = False  # once True, later images are no longer "processed grayscale"


@dataclass(frozen=True)
class StepSpec:
    name: str
    stage: str | None
    technique: str
    opencv: str
    purpose: str
    defaults: dict[str, Any]
    run: Callable[..., None]
    choices: dict[str, tuple] = field(default_factory=dict)


# ---- step implementations ---------------------------------------------------
def _boundary(ctx: Ctx, min_area: float = 0.2):
    quad = geometry.find_document_quad(ctx.bgr, min_area_frac=min_area)
    ctx.info["boundary_found"] = quad is not None
    if quad is not None:
        ctx.bgr = geometry.four_point_transform(ctx.bgr, quad)
    ctx.notes["boundary_found"] = quad is not None


def _gray(ctx: Ctx):
    ctx.img = basic.to_gray(ctx.bgr)


def _denoise(ctx: Ctx, method="median", ksize=3, h=10.0):
    if method == "gaussian":
        ctx.img = basic.gaussian_blur(ctx.img, ksize)
    elif method == "median":
        ctx.img = basic.median_blur(ctx.img, ksize)
    elif method == "nlm":
        ctx.img = basic.nlm_denoise(ctx.img, h)


def _contrast(ctx: Ctx, method="clahe", clip_limit=2.0, tile=8):
    if method == "clahe":
        ctx.img = basic.clahe(ctx.img, clip_limit, tile)
    elif method == "hist_eq":
        ctx.img = basic.hist_equalize(ctx.img)


def _threshold(ctx: Ctx, method="adaptive", value=127, block_size=31, c=10.0):
    ctx.binarized = method != "none"
    if method == "global":
        ctx.img = threshold.global_threshold(ctx.img, value)
    elif method == "otsu":
        ctx.img, t = threshold.otsu_threshold(ctx.img)
        ctx.info["otsu_threshold"] = t
        ctx.notes["otsu_threshold"] = t
    elif method == "adaptive":
        c_used = c if c > 0 else threshold.auto_c(ctx.img)
        ctx.img = threshold.adaptive_threshold(ctx.img, block_size, c_used)
        ctx.notes["c_used"] = round(c_used, 1)


def _deskew(ctx: Ctx, max_angle=15.0):
    theta = deskew.estimate_correction(ctx.img, max_angle)
    ctx.img = deskew.rotate(ctx.img, theta)
    ctx.info["deskew_correction_deg"] = theta
    ctx.notes["rotated_by_deg"] = theta


def _morphology(ctx: Ctx, op="open", ksize=2):
    ctx.img = morphology.apply(ctx.img, op, ksize)


def _resize(ctx: Ctx, min_width=1800, max_width=3000):
    ctx.img, scale = basic.resize_to_width_range(ctx.img, min_width, max_width)
    ctx.notes["scale"] = round(scale, 3)


STEPS: dict[str, StepSpec] = {
    s.name: s
    for s in [
        StepSpec("boundary", None, "Document boundary + perspective transform",
                 "cv2.Canny, cv2.findContours, cv2.approxPolyDP, cv2.getPerspectiveTransform, cv2.warpPerspective",
                 "Finds the page outline in a photo and flattens it to a top-down view.",
                 {"min_area": 0.2}, _boundary),
        StepSpec("gray", "gray", "RGB to grayscale", "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
                 "OCR and thresholding only need brightness; removes coloured stamps and backgrounds.",
                 {}, _gray),
        StepSpec("denoise", "denoised", "Noise removal", "cv2.GaussianBlur / cv2.medianBlur / cv2.fastNlMeansDenoising",
                 "Gaussian smooths grain, median removes salt-and-pepper specks while keeping edges.",
                 {"method": "median", "ksize": 3, "h": 10.0}, _denoise,
                 {"method": ("median", "gaussian", "nlm", "none")}),
        StepSpec("contrast", "denoised", "Contrast enhancement", "cv2.createCLAHE().apply / cv2.equalizeHist",
                 "CLAHE equalizes the histogram per tile, rescuing faded print and uneven lighting.",
                 {"method": "clahe", "clip_limit": 2.0, "tile": 8}, _contrast,
                 {"method": ("clahe", "hist_eq", "none")}),
        StepSpec("threshold", "thresholded", "Thresholding", "cv2.threshold(THRESH_OTSU) / cv2.adaptiveThreshold",
                 "Converts to pure black and white. Otsu picks one global cut-off, adaptive one per neighbourhood.",
                 {"method": "adaptive", "value": 127, "block_size": 31, "c": 0.0}, _threshold,  # c<=0: auto from noise
                 {"method": ("adaptive", "otsu", "global", "none")}),
        StepSpec("deskew", "deskewed", "Deskew (rotation correction)",
                 "projection-profile search, cv2.getRotationMatrix2D, cv2.warpAffine",
                 "Rotates the page so text lines are horizontal; Tesseract degrades quickly with tilt.",
                 {"max_angle": 15.0}, _deskew),
        StepSpec("morphology", "final", "Morphological clean-up", "cv2.morphologyEx(MORPH_OPEN / MORPH_CLOSE)",
                 "Opening removes tiny specks, closing reconnects broken strokes.",
                 {"op": "open", "ksize": 2}, _morphology,
                 {"op": ("open", "close", "erode", "dilate", "none")}),
        StepSpec("resize", "final", "Resize / normalize", "cv2.resize(INTER_CUBIC / INTER_AREA)",
                 "Scales the page so text height suits Tesseract (about 300 DPI).",
                 {"min_width": 1800, "max_width": 3000}, _resize),
    ]
}


@dataclass(frozen=True)
class Recipe:
    name: str
    label: str
    description: str
    steps: tuple[str, ...]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


RECIPES: dict[str, Recipe] = {
    r.name: r
    for r in [
        Recipe("baseline", "R0 Baseline", "Grayscale only. The control for every experiment.", ("gray",)),
        Recipe("basic", "R1 Basic", "Grayscale, Gaussian blur, Otsu threshold.",
               ("gray", "denoise", "threshold"),
               {"denoise": {"method": "gaussian", "ksize": 5}, "threshold": {"method": "otsu"}}),
        Recipe("gray_enhanced", "R2 Enhanced grayscale",
               "Denoise, CLAHE and deskew but no binarization (Tesseract often prefers grayscale).",
               ("gray", "denoise", "contrast", "deskew", "resize")),
        # Default chosen from experiments: on text-on-white scans CLAHE amplifies background
        # noise and hurt OCR on noisy pages, while the adaptive threshold alone handled
        # noise, shadows and skew. CLAHE is kept in R4 for faded documents.
        Recipe("standard", "R3 Standard (default)",
               "Median denoise, adaptive threshold (noise-aware), deskew, light morphology, resize.",
               ("gray", "denoise", "threshold", "deskew", "morphology", "resize")),
        Recipe("standard_clahe", "R4 Standard + CLAHE",
               "R3 with CLAHE contrast enhancement before thresholding. Try it on faded or low-contrast prints.",
               ("gray", "denoise", "contrast", "threshold", "deskew", "morphology", "resize")),
        Recipe("full", "R5 Full (phone photos)",
               "Boundary detection and perspective correction, then the standard recipe.",
               ("boundary", "gray", "denoise", "threshold", "deskew", "morphology", "resize")),
    ]
}
DEFAULT_RECIPE = "standard"


class ParamError(ValueError):
    pass


def resolve_params(recipe: Recipe, user: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Merge defaults < recipe overrides < user overrides, validating names, types, choices."""
    user = user or {}
    for step in user:
        if step not in recipe.steps:
            raise ParamError(f"step '{step}' is not part of recipe '{recipe.name}'")
    out: dict[str, dict[str, Any]] = {}
    for name in recipe.steps:
        spec = STEPS[name]
        params = {**spec.defaults, **recipe.overrides.get(name, {})}
        for key, val in user.get(name, {}).items():
            if key not in spec.defaults:
                raise ParamError(f"unknown parameter '{key}' for step '{name}'")
            default = spec.defaults[key]
            try:
                val = type(default)(val) if not isinstance(default, str) else str(val)
            except (TypeError, ValueError):
                raise ParamError(f"parameter '{key}' of '{name}' must be {type(default).__name__}")
            if key in spec.choices and val not in spec.choices[key]:
                raise ParamError(f"'{key}' of '{name}' must be one of {list(spec.choices[key])}")
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val < 0:
                raise ParamError(f"parameter '{key}' of '{name}' must not be negative")
            params[key] = val
        out[name] = params
    return out


def _is_binary(img: np.ndarray) -> bool:
    return img.ndim == 2 and not np.any((img > 0) & (img < 255))


def _save(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"could not write {path}")


def _small(img: np.ndarray, width: int = EXTRA_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def _make_extras(original: np.ndarray, last_gray: np.ndarray, quad_found: np.ndarray | None, out: Path) -> dict[str, str]:
    """Extra visualizations shown in the Studio: edges, contours, boundary, threshold and
    morphology comparisons. Returns {name: filename}."""
    files: dict[str, str] = {}
    small_bgr = _small(original if original.ndim == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR))
    small_gray = basic.to_gray(small_bgr)

    edges = geometry.canny(small_gray)
    _save(out / "edges.png", cv2.bitwise_not(edges))
    files["edges"] = "edges.png"
    _save(out / "contours.png", geometry.draw_contours(small_bgr, geometry.find_contours(edges)))
    files["contours"] = "contours.png"
    if quad_found is not None:
        scale = small_bgr.shape[1] / original.shape[1]
        overlay = geometry.draw_quad(small_bgr, quad_found * scale)
    else:
        overlay = small_bgr
    _save(out / "boundary.png", overlay)
    files["boundary"] = "boundary.png"

    g = _small(last_gray)
    _save(out / "cmp_global.png", threshold.global_threshold(g))
    otsu, _ = threshold.otsu_threshold(g)
    _save(out / "cmp_otsu.png", otsu)
    _save(out / "cmp_adaptive.png", threshold.adaptive_threshold(g, 31, 0))
    files.update(cmp_global="cmp_global.png", cmp_otsu="cmp_otsu.png", cmp_adaptive="cmp_adaptive.png")
    for op in morphology.OPS:
        name = f"morph_{op}"
        _save(out / f"{name}.png", morphology.apply(otsu, op, 2))
        files[name] = f"{name}.png"
    return files


@dataclass
class Processed:
    final: np.ndarray  # image handed to OCR
    gray_ref: np.ndarray  # last grayscale image before binarization (for quality metrics)
    info: dict[str, Any]
    steps: list[dict[str, Any]]


def process_in_memory(bgr: np.ndarray, recipe_name: str, user_params: dict[str, dict[str, Any]] | None = None) -> Processed:
    """Runs a recipe without saving anything. Used by the evaluation scripts (many images, no disk I/O)."""
    recipe = RECIPES[recipe_name]
    params = resolve_params(recipe, user_params)
    ctx = Ctx(bgr=bgr)
    steps_log = []
    for name in recipe.steps:
        t0 = time.perf_counter()
        ctx.notes = {}
        STEPS[name].run(ctx, **params[name])
        if ctx.img is not None and not ctx.binarized:
            ctx.last_gray = ctx.img
        steps_log.append({"name": name, "ms": (time.perf_counter() - t0) * 1000, "notes": dict(ctx.notes)})
    gray_ref = ctx.last_gray if ctx.last_gray is not None else basic.to_gray(bgr)
    return Processed(ctx.img, gray_ref, ctx.info, steps_log)


def run_pipeline(
    bgr: np.ndarray,
    recipe_name: str,
    user_params: dict[str, dict[str, Any]] | None,
    out_dir: Path,
    url_base: str,
    original_url: str,
    extras: bool = True,
) -> dict[str, Any]:
    recipe = RECIPES[recipe_name]
    params = resolve_params(recipe, user_params)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    t_total = time.perf_counter()

    before = metrics.quality(bgr, with_skew=True)
    ctx = Ctx(bgr=bgr)
    steps_log = []
    for name in recipe.steps:
        spec = STEPS[name]
        t0 = time.perf_counter()
        ctx.notes = {}
        spec.run(ctx, **params[name])
        if ctx.img is not None:
            if not ctx.binarized:
                ctx.last_gray = ctx.img
            if spec.stage:
                ctx.stages[spec.stage] = ctx.img
        steps_log.append({
            "name": name, "technique": spec.technique, "opencv": spec.opencv, "purpose": spec.purpose,
            "params": params[name], "ms": int((time.perf_counter() - t0) * 1000), "notes": ctx.notes,
        })

    final = ctx.img

    stage_infos, prev_url = [], original_url
    for key in STAGE_ORDER:
        if key == "original":
            stage_infos.append({"key": key, "label": STAGE_LABEL[key], "url": original_url, "skipped": False})
            continue
        if key in ctx.stages:  # holds the latest image written to that stage
            _save(out_dir / f"{key}.png", ctx.stages[key])
            prev_url = f"{url_base}/{key}.png?v={stamp}"
            stage_infos.append({"key": key, "label": STAGE_LABEL[key], "url": prev_url, "skipped": False})
        else:  # step not in this recipe: reuse the previous stage image
            stage_infos.append({"key": key, "label": STAGE_LABEL[key], "url": prev_url, "skipped": True})

    reference = ctx.last_gray if ctx.last_gray is not None else basic.to_gray(bgr)
    after = metrics.quality(reference)
    after["skew_deg"] = metrics.skew_degrees(final)
    result: dict[str, Any] = {
        "recipe": recipe.name,
        "label": recipe.label,
        "description": recipe.description,
        "params": params,
        "ran_at": stamp,
        "stages": stage_infos,
        "steps": steps_log,
        "metrics": {
            "before": before,
            "after": after,
            "hist_before": metrics.histogram(basic.to_gray(bgr)),
            "hist_after": metrics.histogram(reference),
        },
        "info": ctx.info,
    }
    if extras:
        quad = geometry.find_document_quad(bgr)
        files = _make_extras(bgr, reference, quad, out_dir)
        result["extras"] = {k: f"{url_base}/{v}?v={stamp}" for k, v in files.items()}
        result["info"]["boundary_detected"] = quad is not None
    result["final_path"] = str(out_dir / "final.png")
    result["total_ms"] = int((time.perf_counter() - t_total) * 1000)
    (out_dir / "meta.json").write_text(json.dumps(result, default=float), encoding="utf-8")
    return result


def recipe_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": r.name, "label": r.label, "description": r.description,
            "default": r.name == DEFAULT_RECIPE,
            "steps": [
                {"name": s, "technique": STEPS[s].technique, "opencv": STEPS[s].opencv,
                 "purpose": STEPS[s].purpose, "params": {**STEPS[s].defaults, **r.overrides.get(s, {})}}
                for s in r.steps
            ],
        }
        for r in RECIPES.values()
    ]
