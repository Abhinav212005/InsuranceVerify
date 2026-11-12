"""Builds the synthetic evaluation dataset (no real personal data).

    python scripts/dataset/build_dataset.py [--per-category 8] [--seed 1]

For every document it writes
    dataset/synthetic/clean/<id>.png            rendered page (TrueType font, ~200 DPI A4)
    dataset/synthetic/truth/<id>.json           category, template, expected text and fields
    dataset/synthetic/manifest.csv              one row per (document, degradation condition) with the true parameters
    dataset/synthetic/degraded/<cond>/<id>.png  only with --write-degraded (they are 2-3 MB each; the evaluation
                                                regenerates them on the fly from the clean page, deterministically)
"""
import argparse
import csv
import json
import random
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import degrade
from app.classification.synthetic import GENERATORS, TEMPLATES

OUT = ROOT.parent / "dataset" / "synthetic"
PAGE_W, PAGE_H = 1654, 2339  # A4 at 200 DPI
FONTS = [  # one per template, so wording AND typography differ; falls back if a font is missing
    ("A", ["arial.ttf", "DejaVuSans.ttf"]),
    ("B", ["calibri.ttf", "arial.ttf", "DejaVuSans.ttf"]),
    ("C", ["times.ttf", "DejaVuSerif.ttf", "arial.ttf"]),
    ("D", ["verdana.ttf", "arial.ttf", "DejaVuSans.ttf"]),
]


def load_font(template: str, size: int):
    for name in dict(FONTS)[template]:
        for base in ("", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"):
            try:
                return ImageFont.truetype(base + name, size), name
            except OSError:
                continue
    return ImageFont.load_default(), "default"


def render(text: str, template: str, rng: random.Random) -> tuple[np.ndarray, str, str]:
    """Draws the text on a white A4 page. Returns (BGR image, ground-truth text, font name)."""
    size = rng.choice([32, 34, 36])
    font, fname = load_font(template, size)
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    lines = []
    for raw in text.split("\n"):
        lines += textwrap.wrap(raw, 70) or [""]
    y = 130 + rng.randint(0, 40)
    for ln in lines:
        draw.text((110, y), ln, fill=(15, 15, 15), font=font)
        y += int(size * 1.7)
    truth_text = " ".join(" ".join(lines).split())
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), truth_text, fname


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write-degraded", action="store_true", help="also save every degraded image to disk (~1.5 GB)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    for sub in ("clean", "truth"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rows, idx = [], 0
    for category, gen in GENERATORS.items():
        for k in range(a.per_category):
            template = TEMPLATES[k % len(TEMPLATES)]
            text, fields = gen(rng, template)
            img, truth_text, font = render(text, template, rng)
            doc_id = f"{category}_{k:02d}"
            cv2.imwrite(str(OUT / "clean" / f"{doc_id}.png"), img)
            (OUT / "truth" / f"{doc_id}.json").write_text(
                json.dumps({"id": doc_id, "category": category, "template": template, "font": font, "text": truth_text, "fields": fields}, indent=1),
                encoding="utf-8",
            )
            for cond in degrade.CONDITIONS:
                out, meta = degrade.apply(cond, img, idx, a.seed)
                if a.write_degraded:
                    folder = OUT / "degraded" / cond
                    folder.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(folder / f"{doc_id}.png"), out)
                rows.append({"id": doc_id, "category": category, "template": template, "condition": cond,
                             "doc_index": idx, "seed": a.seed, "meta": json.dumps(meta),
                             "width": out.shape[1], "height": out.shape[0]})
            idx += 1
        print(f"{category}: {a.per_category} documents")

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{idx} documents x {len(degrade.CONDITIONS)} conditions = {len(rows)} images -> {OUT}")


if __name__ == "__main__":
    main()
