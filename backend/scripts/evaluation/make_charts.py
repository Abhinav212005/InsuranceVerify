"""Turns the experiment CSV/JSON files into charts, summary.json and RESULTS.md.

    python scripts/evaluation/make_charts.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "evaluation_results"
CHARTS = RESULTS / "charts"
COND_ORDER = ["clean", "noise_low", "noise_high", "salt_pepper", "blur_low", "blur_high", "rot_3", "rot_8", "rot_15",
              "lowres_150", "lowres_100", "shadow", "faded", "jpeg_low", "photo"]
CATS = ["insurance_policy", "claim_form", "identity_document", "medical_bill", "repair_invoice", "other"]
PALETTE = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#64748b", "#db2777", "#65a30d", "#0ea5e9"]

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def read(name):
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        return None
    out = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            out.append(r)
    return out


def gmean(rows, keys, val):
    g = defaultdict(list)
    for r in rows:
        x = r.get(val)
        if isinstance(x, float) and not np.isnan(x):
            g[tuple(r[k] for k in keys)].append(x)
    return {k: float(np.mean(v)) for k, v in g.items()}


def order(items, ref):
    return [x for x in ref if x in items] + [x for x in items if x not in ref]


def save(fig, name):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS / name)
    plt.close(fig)
    print("  chart", name)


def pct(x):
    return f"{x * 100:.1f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


SUMMARY: dict = {}
MD: list[str] = []


# ---- 1 ablation ---------------------------------------------------------------------------------------------
def ch_ablation():
    rows = read("exp1_ablation")
    if not rows:
        return
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    conds = order({r["condition"] for r in rows}, COND_ORDER)
    cer = gmean(rows, ["variant"], "cer")
    wer = gmean(rows, ["variant"], "wer")
    cell = gmean(rows, ["variant", "condition"], "cer")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [cer[(x,)] for x in variants]
    bars = ax.barh(variants[::-1], vals[::-1], color=["#4f46e5" if "default" in x or "full" in x else "#94a3b8" for x in variants[::-1]])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals[::-1]], padding=3)
    ax.set_xlabel("mean character error rate (lower is better)")
    ax.set_title("Experiment 1: effect of each preprocessing step (all 15 degradations)")
    save(fig, "01_ablation_bar.png")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    m = np.array([[min(cell.get((vn, c), np.nan), 1.0) for c in conds] for vn in variants])
    im = ax.imshow(m, cmap="RdYlGn_r", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right")
    ax.set_yticks(range(len(variants)), variants)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="CER")
    ax.set_title("CER by preprocessing variant and degradation")
    save(fig, "01_ablation_heatmap.png")
    SUMMARY["ablation"] = {"n_images": len(rows) // len(variants), "mean_cer": {x: cer[(x,)] for x in variants},
                           "mean_wer": {x: wer[(x,)] for x in variants},
                           "cer_by_condition": {x: {c: cell.get((x, c)) for c in conds} for x in variants}}
    MD.append("## 1. Preprocessing ablation\n\nCumulative pipelines, OCR error against the known text. "
              f"{len(rows) // len(variants)} images per row (all 15 degradation conditions).\n\n" +
              table(["Variant", "mean CER", "mean WER"], [[x, f"{cer[(x,)]:.4f}", f"{wer[(x,)]:.4f}"] for x in variants]) +
              "\n\n![ablation](charts/01_ablation_bar.png)\n\n![heatmap](charts/01_ablation_heatmap.png)\n")


# ---- generic grouped bar --------------------------------------------------------------------------------------------
def grouped(ax, groups, series, data, ylabel, fmt="{:.2f}", annotate=False):
    w = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [data.get((s, g), np.nan) for g in groups]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=s, color=PALETTE[i % len(PALETTE)])
        if annotate:
            ax.bar_label(bars, labels=[fmt.format(v) if not np.isnan(v) else "" for v in vals], fontsize=6, padding=1)
    ax.set_xticks(x, groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False)


def ch_threshold():
    rows = read("exp2_threshold")
    if not rows:
        return
    conds = order({r["condition"] for r in rows}, COND_ORDER)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    cell = gmean(rows, ["variant", "condition"], "cer")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    grouped(ax, conds, variants, {(v, c): cell.get((v, c), np.nan) for v in variants for c in conds}, "mean CER", annotate=True)
    ax.set_title("Experiment 2: thresholding method under different degradations")
    save(fig, "02_threshold.png")
    SUMMARY["threshold"] = {v: {c: cell.get((v, c)) for c in conds} for v in variants}
    MD.append("## 2. Thresholding\n\nMean CER. Global thresholding breaks under shadows; Otsu is excellent on noise but cannot handle uneven light; adaptive handles both.\n\n" +
              table(["Method"] + conds, [[v] + [f"{cell.get((v, c), float('nan')):.3f}" for c in conds] for v in variants]) + "\n\n![threshold](charts/02_threshold.png)\n")


def ch_denoise():
    rows = read("exp3_denoise")
    if not rows:
        return
    conds = order({r["condition"] for r in rows}, COND_ORDER)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, label) in zip(axes, [("psnr", "PSNR (dB, higher better)"), ("ssim", "SSIM (higher better)"), ("cer", "OCR CER (lower better)")]):
        cell = gmean(rows, ["variant", "condition"], key)
        grouped(ax, conds, variants, {(v, c): cell.get((v, c), np.nan) for v in variants for c in conds}, label)
        ax.set_title(label)
    axes[0].legend(fontsize=6.5, frameon=False)
    for ax in axes[1:]:
        ax.get_legend().remove()
    fig.suptitle("Experiment 3: denoising methods (image fidelity vs the clean original, and OCR error)")
    save(fig, "03_denoise.png")
    ps, ss, ce = (gmean(rows, ["variant"], k) for k in ("psnr", "ssim", "cer"))
    SUMMARY["denoise"] = {v: {"psnr": ps.get((v,)), "ssim": ss.get((v,)), "cer": ce.get((v,))} for v in variants}
    MD.append("## 3. Denoising\n\nPSNR/SSIM compare the denoised grayscale page with the clean original; CER is after thresholding and OCR.\n\n" +
              table(["Method", "PSNR dB", "SSIM", "CER"], [[v, f"{ps.get((v,), float('nan')):.2f}", f"{ss.get((v,), float('nan')):.3f}", f"{ce.get((v,), float('nan')):.4f}"] for v in variants]) +
              "\n\n![denoise](charts/03_denoise.png)\n")


def ch_skew():
    rows = read("exp4_skew")
    if not rows:
        return
    est = [r for r in rows if r["variant"] == "with deskew"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    t = [r["true_angle"] for r in est]
    e = [r["est_angle"] for r in est]
    axes[0].scatter(t, e, alpha=0.6, color="#4f46e5")
    lim = max(abs(np.array(t + e))) + 1
    axes[0].plot([-lim, lim], [-lim, lim], "--", color="#94a3b8")
    err = np.mean([r["angle_error"] for r in est])
    axes[0].set_xlabel("true skew (degrees)")
    axes[0].set_ylabel("estimated skew (degrees)")
    axes[0].set_title(f"Skew estimation (mean absolute error {err:.2f} deg)")
    conds = order({r["condition"] for r in rows}, COND_ORDER)
    cell = gmean(rows, ["variant", "condition"], "cer")
    grouped(axes[1], conds, ["with deskew", "without deskew"], {(v, c): cell.get((v, c), np.nan) for v in ("with deskew", "without deskew") for c in conds}, "mean CER", annotate=True)
    axes[1].set_title("OCR error with and without deskew")
    fig.suptitle("Experiment 4: skew correction")
    save(fig, "04_skew.png")
    by_cond = {c: float(np.mean([r["angle_error"] for r in est if r["condition"] == c])) for c in conds if any(r["condition"] == c for r in est)}
    SUMMARY["skew"] = {"mean_angle_error_deg": float(err), "max_angle_error_deg": float(max(r["angle_error"] for r in est)),
                       "angle_error_by_condition": by_cond, "cer": {v: {c: cell.get((v, c)) for c in conds} for v in ("with deskew", "without deskew")}}
    MD.append(f"## 4. Skew correction\n\nMean angle error **{err:.2f} degrees** (max {SUMMARY['skew']['max_angle_error_deg']:.2f}).\n\n" +
              table(["Condition", "angle error (deg)", "CER with deskew", "CER without"], [[c, f"{by_cond.get(c, float('nan')):.2f}", f"{cell.get(('with deskew', c), float('nan')):.4f}", f"{cell.get(('without deskew', c), float('nan')):.4f}"] for c in conds]) +
              "\n\n![skew](charts/04_skew.png)\n")


def ch_resolution():
    rows = read("exp5_resolution")
    if not rows:
        return
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    scales = sorted({r["scale"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, ax, label in (("cer", axes[0], "mean CER"), ("ocr_conf", axes[1], "mean OCR confidence (%)")):
        cell = gmean(rows, ["variant", "scale"], key)
        for i, vname in enumerate(variants):
            ax.plot([200 * s for s in scales], [cell.get((vname, s), np.nan) for s in scales], "o-", label=vname, color=PALETTE[i])
        ax.set_xlabel("scan resolution (DPI)")
        ax.set_ylabel(label)
        ax.invert_xaxis()
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Experiment 5: OCR vs scan resolution")
    save(fig, "05_resolution.png")
    cer = gmean(rows, ["variant", "scale"], "cer")
    SUMMARY["resolution"] = {v: {f"{int(200 * s)} dpi": cer.get((v, s)) for s in scales} for v in variants}
    MD.append("## 5. Resolution\n\nMean CER by simulated scan resolution (200 DPI is the full-size page).\n\n" +
              table(["Variant"] + [f"{int(200 * s)} DPI" for s in scales], [[v] + [f"{cer.get((v, s), float('nan')):.3f}" for s in scales] for v in variants]) + "\n\n![resolution](charts/05_resolution.png)\n")


def ch_photo():
    rows = read("exp6_photo")
    if not rows:
        return
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    cer = gmean(rows, ["variant"], "cer")
    raw = [r for r in rows if r["variant"] == "raw image"]
    found = float(np.mean([r["found"] for r in raw]))
    iou = float(np.mean([r["iou"] for r in raw if r["found"]])) if any(r["found"] for r in raw) else float("nan")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(variants, [cer[(v,)] for v in variants], color=PALETTE[: len(variants)])
    ax.set_ylabel("mean CER")
    ax.set_title(f"Experiment 6: simulated phone photos (page found in {pct(found)}, mean IoU {iou:.2f})")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save(fig, "06_photo.png")
    SUMMARY["photo"] = {"boundary_found_rate": found, "mean_iou": iou, "cer": {v: cer[(v,)] for v in variants}, "n": len(raw)}
    MD.append(f"## 6. Phone photos (boundary detection and perspective)\n\nPage outline found in **{pct(found)}** of {len(raw)} photos; mean IoU with the true page **{iou:.2f}**.\n\n" +
              table(["Variant", "mean CER"], [[v, f"{cer[(v,)]:.4f}"] for v in variants]) + "\n\n![photo](charts/06_photo.png)\n")


def ch_psm():
    rows = read("exp7_psm")
    if not rows:
        return
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    cats = order({r["category"] for r in rows}, CATS)
    cell = gmean(rows, ["variant", "category"], "cer")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    grouped(ax, cats, variants, {(v, c): cell.get((v, c), np.nan) for v in variants for c in cats}, "mean CER")
    ax.set_title("Experiment 7: Tesseract page segmentation mode by document category")
    save(fig, "07_psm.png")
    allv = gmean(rows, ["variant"], "cer")
    SUMMARY["psm"] = {"overall": {v: allv[(v,)] for v in variants}, "by_category": {v: {c: cell.get((v, c)) for c in cats} for v in variants}}
    MD.append("## 7. Page segmentation mode\n\n" + table(["Mode"] + cats, [[v] + [f"{cell.get((v, c), float('nan')):.4f}" for c in cats] for v in variants]) + "\n\n![psm](charts/07_psm.png)\n")


def ch_e2e():
    rows = read("exp8_9_e2e")
    if not rows:
        return
    conds = order({r["condition"] for r in rows}, COND_ORDER)
    methods = ["keyword", "ml", "hybrid"]
    acc = {(m, c): np.mean([r[f"pred_{m}"] == r["true_category"] for r in rows if r["condition"] == c]) for m in methods for c in conds}
    overall = {m: float(np.mean([r[f"pred_{m}"] == r["true_category"] for r in rows])) for m in methods}
    fig, ax = plt.subplots(figsize=(11, 3.8))
    grouped(ax, conds, methods, acc, "classification accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Experiment 8: document classification on OCR text from degraded images")
    save(fig, "08_classification.png")
    # confusion matrix (hybrid)
    idx = {c: i for i, c in enumerate(CATS)}
    cm = np.zeros((6, 6), int)
    for r in rows:
        cm[idx[r["true_category"]], idx.get(r["pred_hybrid"], 5)] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(6), [c.replace("_", "\n") for c in CATS], fontsize=6.5)
    ax.set_yticks(range(6), [c.replace("_", " ") for c in CATS], fontsize=7)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Hybrid classifier, all conditions")
    save(fig, "08_confusion.png")
    # template-held-out benchmark from eval_classifier.py
    held = {}
    p = RESULTS / "classifier_eval.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        held = {k: v["accuracy"] for k, v in j["results"].items()}
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ks = ["keyword", "ml", "hybrid"]
        grouped(ax, ["clean", "noisy"], ks, {(m, t): held.get(f"{m}/{t}", np.nan) for m in ks for t in ("clean", "noisy")}, "accuracy", annotate=True)
        ax.set_ylim(0, 1.1)
        ax.set_title("Unseen layout (train on templates A-C, test on D)")
        save(fig, "08_classifier_heldout.png")
    # per-class precision/recall/F1 for hybrid (macro)
    prf = {}
    for c in CATS:
        tp = sum(r["true_category"] == c and r["pred_hybrid"] == c for r in rows)
        fp = sum(r["true_category"] != c and r["pred_hybrid"] == c for r in rows)
        fn = sum(r["true_category"] == c and r["pred_hybrid"] != c for r in rows)
        p_, r_ = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        prf[c] = {"precision": p_, "recall": r_, "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0}
    SUMMARY["classification"] = {"overall_accuracy": overall, "by_condition": {m: {c: float(acc[(m, c)]) for c in conds} for m in methods},
                                 "hybrid_per_class": prf, "heldout_template": held, "n": len(rows)}
    MD.append("## 8. Document classification\n\nEnd to end: the page is processed and OCR'd, then classified from the recognised text. " +
              f"{len(rows)} images.\n\n" + table(["Method", "overall accuracy"], [[m, pct(overall[m])] for m in methods]) +
              "\n\nHybrid per-class results:\n\n" + table(["Category", "precision", "recall", "F1"], [[c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}"] for c, v in prf.items()]) +
              "\n\n![classification](charts/08_classification.png)\n\n![confusion](charts/08_confusion.png)\n\n![heldout](charts/08_classifier_heldout.png)\n")

    # ---- 9 field extraction
    def acc_fields(tag, cat=None, cond=None):
        ok = tot = 0
        for r in rows:
            if (cat and r["true_category"] != cat) or (cond and r["condition"] != cond) or r["true_category"] == "other":
                continue
            d = json.loads(r[f"fields_{tag}"])
            ok += sum(d.values())
            tot += len(d)
        return ok / tot if tot else np.nan

    cats = CATS[:5]
    heat = np.array([[acc_fields("oracle", c, k) for k in conds] for c in cats])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right")
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="field accuracy")
    ax.set_title("Experiment 9: field extraction accuracy (true category) by document type and degradation")
    save(fig, "09_fields_heatmap.png")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    grouped(ax, conds, ["oracle", "pipeline"], {(t, c): acc_fields(t, None, c) for t in ("oracle", "pipeline") for c in conds}, "field accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Field accuracy: with the true category (oracle) vs the category the system predicted")
    save(fig, "09_fields_bar.png")
    perfield = defaultdict(list)
    for r in rows:
        if r["true_category"] == "other":
            continue
        for k, v in json.loads(r["fields_oracle"]).items():
            perfield[f"{r['true_category']}.{k}"].append(v)
    pf = {k: float(np.mean(v)) for k, v in sorted(perfield.items())}
    weakest = sorted(pf.items(), key=lambda kv: kv[1])[:8]
    SUMMARY["fields"] = {"overall_oracle": acc_fields("oracle"), "overall_pipeline": acc_fields("pipeline"),
                         "by_category": {c: acc_fields("oracle", c) for c in cats}, "by_condition": {k: acc_fields("oracle", None, k) for k in conds},
                         "per_field": pf, "weakest": weakest}
    MD.append(f"## 9. Field extraction\n\nOverall accuracy **{pct(SUMMARY['fields']['overall_oracle'])}** with the true category, **{pct(SUMMARY['fields']['overall_pipeline'])}** when the system's own predicted category is used.\n\n" +
              table(["Category", "field accuracy"], [[c, pct(acc_fields('oracle', c))] for c in cats]) + "\n\nWeakest fields:\n\n" +
              table(["Field", "accuracy"], [[k, pct(v)] for k, v in weakest]) + "\n\n![fields](charts/09_fields_heatmap.png)\n\n![fields bar](charts/09_fields_bar.png)\n")


def ch_timing():
    rows = read("exp10_timing")
    if not rows:
        return
    res = list(dict.fromkeys(r["resolution"] for r in rows))
    stages = [("dip_gray_ms", "grayscale"), ("dip_denoise_ms", "denoise"), ("dip_threshold_ms", "threshold"), ("dip_deskew_ms", "deskew"),
              ("dip_morphology_ms", "morphology"), ("dip_resize_ms", "resize"), ("ocr_ms", "OCR (Tesseract)"), ("classify_ms", "classify"),
              ("extract_ms", "extract"), ("validate_ms", "validate")]
    means = {x: {k: float(np.mean([r.get(k, 0) or 0 for r in rows if r["resolution"] == x])) for k, _ in stages} for x in res}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = np.zeros(len(res))
    for i, (k, label) in enumerate(stages):
        vals = np.array([means[x][k] for x in res])
        ax.barh(res, vals, left=left, label=label, color=PALETTE[i % len(PALETTE)])
        left += vals
    ax.set_xlabel("milliseconds per page")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Experiment 10: processing time per stage")
    save(fig, "10_timing.png")
    SUMMARY["timing"] = {x: {**{lab: means[x][k] for k, lab in stages}, "total_ms": sum(means[x].values())} for x in res}
    MD.append("## 10. Processing time\n\nMean milliseconds per page (single process, no contention).\n\n" +
              table(["Stage"] + res, [[lab] + [f"{means[x][k]:.0f}" for x in res] for k, lab in stages] + [["**total**"] + [f"**{sum(means[x].values()):.0f}**" for x in res]]) + "\n\n![timing](charts/10_timing.png)\n")


def ch_validation():
    p = RESULTS / "exp11_validation.json"
    if not p.exists():
        return
    j = json.loads(p.read_text(encoding="utf-8"))
    codes = [c for c, v in j["per_code"].items() if v["recall"] is not None or v["precision"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    y = np.arange(len(codes))
    axes[0].barh(y - 0.2, [j["per_code"][c]["precision"] or 0 for c in codes], 0.4, label="precision", color="#4f46e5")
    axes[0].barh(y + 0.2, [j["per_code"][c]["recall"] or 0 for c in codes], 0.4, label="recall", color="#059669")
    axes[0].set_yticks(y, codes, fontsize=7)
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Issue detection per rule")
    ds = sorted(j["by_defect"].items())
    axes[1].barh([d for d, _ in ds], [v["status_accuracy"] for _, v in ds], color="#d97706")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title(f"Review status accuracy (overall {j['status_accuracy']:.3f})")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.suptitle("Experiment 11: validation engine on documents with injected defects")
    save(fig, "11_validation.png")
    SUMMARY["validation"] = j
    MD.append(f"## 11. Validation\n\n{j['cases']} documents with known defects (and clean controls). Review-status accuracy **{pct(j['status_accuracy'])}**.\n\n" +
              table(["Rule", "TP", "FP", "FN", "precision", "recall"], [[c, v["tp"], v["fp"], v["fn"], "-" if v["precision"] is None else f"{v['precision']:.2f}", "-" if v["recall"] is None else f"{v['recall']:.2f}"] for c, v in j["per_code"].items()]) +
              "\n\n![validation](charts/11_validation.png)\n")


def main() -> None:
    for fn in (ch_ablation, ch_threshold, ch_denoise, ch_skew, ch_resolution, ch_photo, ch_psm, ch_e2e, ch_timing, ch_validation):
        fn()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float), encoding="utf-8")
    head = ("# Evaluation results\n\nGenerated by `scripts/evaluation/make_charts.py` from the CSV files in this folder "
            "(produced by `run_experiments.py`). Dataset: 48 synthetic documents (6 categories x 8, four layouts and four fonts) "
            "under 15 controlled degradations; no real personal data.\n\n"
            "**Limitations.** The documents are synthetic and rendered with clean TrueType fonts, and the extraction/classification "
            "rules were written by the project authors, so these numbers measure *this system on this benchmark*; they are not a "
            "claim about real-world accuracy. Use public scanned documents for the final report.\n\n")
    (RESULTS / "RESULTS.md").write_text(head + "\n".join(MD), encoding="utf-8")
    print("wrote summary.json and RESULTS.md")


if __name__ == "__main__":
    main()
