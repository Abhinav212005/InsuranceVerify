"""Runs the evaluation experiments and writes CSV/JSON files to evaluation_results/.

    python scripts/evaluation/run_experiments.py --exp all [--docs 3] [--workers 10]
    python scripts/evaluation/run_experiments.py --exp ablation,threshold

Experiments (numbers follow the project plan):
  1 ablation    cumulative effect of each preprocessing technique on OCR error (CER/WER)
  2 threshold   global vs Otsu vs adaptive thresholding under different degradations
  3 denoise     none / Gaussian / median / non-local means (PSNR, SSIM and CER)
  4 skew        deskew angle error and the effect of deskew on CER
  5 resolution  OCR error vs scan resolution, with and without upscaling
  6 photo       page boundary detection (success, IoU) and CER on simulated phone photos
  7 psm         Tesseract page-segmentation modes
  8/9 e2e       end-to-end: OCR text -> classification (keyword/ML/hybrid) and field extraction
  10 timing     processing time per stage
  11 validation precision/recall of the validation rules on documents with injected defects
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (also puts backend/ and scripts/dataset on sys.path)

from app.classification.classifier import classify  # noqa: E402
from app.extraction.extractors import extract  # noqa: E402
from app.extraction.schema import specs  # noqa: E402

ALL_CONDITIONS = list(__import__("degrade").CONDITIONS)


def tasks_for(rows, variants, **extra):
    return [{"row": r, "variant": name, "recipe": recipe, **extra, **kw} for r in rows for name, recipe, kw in variants]


def v(name, recipe, **kw):
    return (name, recipe, kw)


# ---------------------------------------------------------------------------------------------------
def exp_ablation(a, rows):
    sel = C.pick(rows, ALL_CONDITIONS, a.docs)
    variants = [v(name, recipe) for name, recipe in C.ABLATION]
    res = C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "ablation")
    C.write_csv("exp1_ablation", res)


def exp_threshold(a, rows):
    sel = C.pick(rows, ["clean", "noise_high", "shadow", "faded", "jpeg_low"], a.docs)
    variants = [v("raw grayscale (no threshold)", "gray_only"), v("global (T=127)", "thr_global"), v("Otsu", "thr_otsu"), v("adaptive", "thr_adaptive")]
    C.write_csv("exp2_threshold", C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "threshold"))


def exp_denoise(a, rows):
    sel = C.pick(rows, ["noise_low", "noise_high", "salt_pepper"], a.docs)
    variants = [v("none", "den_none"), v("Gaussian 5x5", "den_gauss"), v("median 3", "den_median3"), v("median 5", "den_median5"), v("non-local means", "den_nlm")]
    C.write_csv("exp3_denoise", C.parallel(C.run_task, tasks_for(sel, variants, quality=True), a.workers, "denoise"))


def exp_skew(a, rows):
    sel = C.pick(rows, ["clean", "rot_3", "rot_8", "rot_15"], a.docs)
    variants = [v("with deskew", "standard", skew=True), v("without deskew", "no_deskew")]
    C.write_csv("exp4_skew", C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "skew"))


def exp_resolution(a, rows):
    sel = C.pick(rows, ["clean"], a.docs)
    variants = []
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25):
        for name, recipe in (("grayscale only", "gray_only"), ("grayscale + upscale", "gray_upscale"), ("default pipeline", "standard")):
            variants.append(v(name, recipe, scale=scale))
    C.write_csv("exp5_resolution", C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "resolution"))


def exp_photo(a, rows):
    sel = C.pick(rows, ["photo"], 8)
    variants = [v("raw image", None, boundary=True), v("R3 default", "standard"), v("R5 full (boundary + perspective)", "full")]
    C.write_csv("exp6_photo", C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "photo"))


def exp_psm(a, rows):
    sel = C.pick(rows, ["clean", "noise_low", "blur_low"], a.docs)
    variants = [v(f"PSM {p}", "standard", psm=p) for p in (3, 4, 6, 11)]
    C.write_csv("exp7_psm", C.parallel(C.run_task, tasks_for(sel, variants), a.workers, "psm"))


# ---- 8 + 9: end to end ----------------------------------------------------------------------------------
def _norm(x):
    return None if x is None else re.sub(r"[^a-z0-9]", "", str(x).lower())


def e2e_task(spec):
    row = spec["row"]
    truth = C.truth_of(row["id"])
    out = C.run_task({**spec, "text": True})
    text = out.pop("text")
    pages = [{"page": 1, "text": text, "words": [], "avg_confidence": out["ocr_conf"]}]
    for m in ("keyword", "ml", "hybrid"):
        out[f"pred_{m}"] = classify(text, m)["category"]
    pred = out["pred_hybrid"]
    for tag, cat in (("oracle", row["category"]), ("pipeline", pred)):
        got = {f.name: f.value for f in extract(cat, [dict(p) for p in pages])} if cat != "other" else {}
        results = {k: int(_norm(got.get(k)) == _norm(val)) for k, val in truth["fields"].items()} if cat == row["category"] or tag == "oracle" else {k: 0 for k in truth["fields"]}
        out[f"fields_{tag}"] = json.dumps(results)
    out["true_category"] = row["category"]
    return out


def exp_e2e(a, rows):
    sel = C.pick(rows, ALL_CONDITIONS, a.e2e_docs)
    tasks = [{"row": r, "variant": "e2e", "recipe": "standard"} for r in sel]
    C.write_csv("exp8_9_e2e", C.parallel(e2e_task, tasks, a.workers, "e2e"))


# ---- 10: timing (sequential on purpose: no contention between processes) ---------------------------------
def exp_timing(a, rows):
    import cv2

    from app.services import classification_service  # noqa: F401
    from app.validation import engine
    from app.validation.rules import Ctx, FieldView
    from datetime import date

    sel = C.pick(rows, ["clean"], 2)
    out = []
    for r in sel:
        for label, scale in (("200 DPI (1654 px)", 1.0), ("300 DPI (2481 px)", 1.5)):
            img = C.load_image(r, scale)
            t0 = time.perf_counter()
            p = C.process_in_memory(img, "standard")
            dip = (time.perf_counter() - t0) * 1000
            text, conf, ocr_ms = C.ocr(p.final, 3)
            pages = [{"page": 1, "text": text, "words": [], "avg_confidence": conf}]
            t = time.perf_counter(); res = classify(text, "hybrid"); cls_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); fields = extract(res["category"], pages); ext_ms = (time.perf_counter() - t) * 1000
            ctx = Ctx(res["category"], specs(res["category"]), {f.name: FieldView(f.name, f.value, f.confidence) for f in fields}, date.today())
            t = time.perf_counter(); engine.validate(ctx); val_ms = (time.perf_counter() - t) * 1000
            row = {"id": r["id"], "resolution": label, "width": img.shape[1], "ocr_ms": ocr_ms, "classify_ms": cls_ms,
                   "extract_ms": ext_ms, "validate_ms": val_ms, **{f"dip_{s['name']}_ms": s["ms"] for s in p.steps}, "dip_total_ms": dip}
            out.append(row)
            print(f"  timing {r['id']} {label}: dip {dip:.0f} ms, ocr {ocr_ms:.0f} ms", flush=True)
    C.write_csv("exp10_timing", out)


# ---- 11: validation rules on documents with injected defects ----------------------------------------------------
def _policy_text(rng):
    from app.classification.synthetic import policy
    text, _ = policy(rng, "A")
    return re.sub(r"Period of Insurance: .*", "Period of Insurance: 01/04/2027 to 31/03/2028", text)


def _claim_text(rng):
    from app.classification.synthetic import claim
    text, _ = claim(rng, "A")
    return re.sub(r"Date of Incident: .*", "Date of Incident: 12/05/2025", text)


DEFECTS = [  # name, base, edit(text) -> text, expected rule codes
    ("control (no defect)", "policy", lambda t: t, set()),
    ("control claim (no defect)", "claim", lambda t: t, set()),
    ("missing policy number", "policy", lambda t: re.sub(r"Policy No: .*\n", "", t), {"POLICY_NUMBER_MISSING"}),
    ("missing premium", "policy", lambda t: re.sub(r"Premium: .*\n", "\n", t), {"MISSING_FIELD"}),
    ("expired policy", "policy", lambda t: re.sub(r"Period of Insurance: .*", "Period of Insurance: 01/01/2019 to 31/12/2019", t), {"POLICY_EXPIRED"}),
    ("impossible date (31/02)", "policy", lambda t: re.sub(r"Period of Insurance: \S+", "Period of Insurance: 31/02/2027", t), {"INVALID_DATE"}),
    ("expiry before start", "policy", lambda t: re.sub(r"Period of Insurance: .*", "Period of Insurance: 01/04/2030 to 01/04/2029", t), {"DATE_ORDER"}),
    ("zero premium", "policy", lambda t: re.sub(r"Premium: Rs\. .*", "Premium: Rs. 0", t), {"INVALID_AMOUNT"}),
    ("absurd premium", "policy", lambda t: re.sub(r"Premium: Rs\. .*", "Premium: Rs. 99,99,99,99,999", t), {"INVALID_AMOUNT"}),
    ("missing claim number", "claim", lambda t: re.sub(r"Claim No: .*\n", "", t), {"CLAIM_NUMBER_MISSING"}),
    ("incident date in the future", "claim", lambda t: re.sub(r"Date of Incident: .*", "Date of Incident: 05/05/2031", t), {"FUTURE_DATE"}),
]
SEVERE = ("error", "warning")


def exp_validation(a, rows):
    import random
    from datetime import date

    from app.validation import engine
    from app.validation.rules import Ctx, FieldView

    rng = random.Random(7)
    cases = []
    per = 30
    for name, base, edit, expected in DEFECTS:
        for _ in range(per):
            text = edit(_policy_text(rng) if base == "policy" else _claim_text(rng))
            cat = "insurance_policy" if base == "policy" else "claim_form"
            fields = {f.name: FieldView(f.name, f.value, f.confidence) for f in extract(cat, [{"page": 1, "text": text, "words": [], "avg_confidence": 95.0}])}
            out = engine.validate(Ctx(cat, specs(cat), fields, date.today(), ocr_confidence=95.0, class_confidence=0.9))
            got = {i.code for i in out.issues if i.severity in SEVERE}
            want_status = "incomplete" if any(c in expected for c in ("POLICY_NUMBER_MISSING", "MISSING_FIELD", "CLAIM_NUMBER_MISSING")) else ("needs_review" if expected else "valid")
            cases.append({"defect": name, "expected": sorted(expected), "got": sorted(got), "status": out.status, "want_status": want_status})
    # low-confidence reading: the same clean documents, but OCR reports poor quality
    for _ in range(per):
        text = _policy_text(rng)
        fields = {f.name: FieldView(f.name, f.value, f.confidence * 0.5) for f in extract("insurance_policy", [{"page": 1, "text": text, "words": [], "avg_confidence": 45.0}])}
        out = engine.validate(Ctx("insurance_policy", specs("insurance_policy"), fields, date.today(), ocr_confidence=45.0, class_confidence=0.9))
        got = {i.code for i in out.issues if i.severity in SEVERE}
        cases.append({"defect": "poor OCR quality", "expected": ["LOW_CONFIDENCE_FIELD", "LOW_OCR_CONFIDENCE"], "got": sorted(got), "status": out.status, "want_status": "needs_review"})

    codes = sorted({c for k in cases for c in k["expected"] + k["got"]})
    per_code = {}
    for code in codes:
        tp = sum(code in k["expected"] and code in k["got"] for k in cases)
        fp = sum(code not in k["expected"] and code in k["got"] for k in cases)
        fn = sum(code in k["expected"] and code not in k["got"] for k in cases)
        per_code[code] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    by_defect = {}
    for d in {k["defect"] for k in cases}:
        ks = [k for k in cases if k["defect"] == d]
        by_defect[d] = {"cases": len(ks), "status_accuracy": sum(k["status"] == k["want_status"] for k in ks) / len(ks),
                        "exact_issue_match": sum(set(k["expected"]) <= set(k["got"]) and set(k["got"]) <= set(k["expected"]) | {"LOW_CONFIDENCE_FIELD"} for k in ks) / len(ks)}
    summary = {"cases": len(cases), "status_accuracy": sum(k["status"] == k["want_status"] for k in cases) / len(cases), "per_code": per_code, "by_defect": by_defect}
    C.RESULTS.mkdir(exist_ok=True)
    (C.RESULTS / "exp11_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  validation: {len(cases)} cases, status accuracy {summary['status_accuracy']:.3f}")


EXPERIMENTS = {
    "ablation": exp_ablation, "threshold": exp_threshold, "denoise": exp_denoise, "skew": exp_skew,
    "resolution": exp_resolution, "photo": exp_photo, "psm": exp_psm, "e2e": exp_e2e,
    "timing": exp_timing, "validation": exp_validation,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--docs", type=int, default=3, help="documents per category for the OCR experiments")
    ap.add_argument("--e2e-docs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    names = list(EXPERIMENTS) if a.exp == "all" else a.exp.split(",")
    rows = C.read_manifest()
    for n in names:
        t0 = time.time()
        print(f"== {n}", flush=True)
        EXPERIMENTS[n](a, rows)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
