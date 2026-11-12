"""Field-extraction accuracy on (a) a synthetic corpus with ground truth and (b) hand-written documents.

    python scripts/evaluation/eval_extraction.py [--per-template 30] [--noise 0.04] [--save]

A field counts as correct when the extracted value equals the truth after normalisation
(dates are ISO, amounts have 2 decimals, text/names compare case- and space-insensitively).
A field the document does not contain must come back empty. Run on clean text and on
text with simulated OCR errors.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.classification.synthetic import generate
from app.extraction.extractors import extract

OUT_DIR = ROOT.parent / "evaluation_results"


def norm(v):
    return None if v is None else re.sub(r"[^a-z0-9]", "", str(v).lower())


def score(samples, label_of=lambda s: s.label):
    per_field: dict[str, list[int]] = defaultdict(list)
    per_cat: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        if not s.truth:
            continue
        fields = {f.name: f.value for f in extract(label_of(s), [{"page": 1, "text": s.text, "words": [], "avg_confidence": 95.0}])}
        for name, truth in s.truth.items():
            ok = int(norm(fields.get(name)) == norm(truth))
            per_field[f"{label_of(s)}.{name}"].append(ok)
            per_cat[label_of(s)].append(ok)
    return per_field, per_cat


def summarize(title, samples):
    pf, pc = score(samples)
    total = [x for v in pc.values() for x in v]
    print(f"\n{title}: {len(samples)} documents, {len(total)} fields, accuracy {sum(total) / len(total):.3f}")
    for cat, v in sorted(pc.items()):
        print(f"  {cat:20s} {sum(v) / len(v):.3f}  ({len(v)} fields)")
    return {"accuracy": sum(total) / len(total), "fields": len(total),
            "per_category": {c: sum(v) / len(v) for c, v in pc.items()},
            "per_field": {f: sum(v) / len(v) for f, v in sorted(pf.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=30)
    ap.add_argument("--noise", type=float, default=0.04, help="max OCR character error rate for the noisy run")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    out = {}
    out["synthetic_clean"] = summarize("synthetic, clean text", generate(a.per_template, noise=0.0, seed=a.seed))
    out["synthetic_noisy"] = summarize(f"synthetic, OCR noise up to {a.noise:.0%}", generate(a.per_template, noise=a.noise, seed=a.seed))

    from handwritten_docs import DOCS
    from handwritten_truth import TRUTH
    class S:  # tiny adapter
        def __init__(self, label, text, truth): self.label, self.text, self.truth = label, text, truth
    hand = [S(l, t, TRUTH[i]) for i, (l, t) in enumerate(DOCS)]
    out["handwritten"] = summarize("hand-written documents", hand)
    worst = sorted(out["synthetic_noisy"]["per_field"].items(), key=lambda kv: kv[1])[:8]
    print("\nweakest fields (synthetic, noisy):")
    for f, v in worst:
        print(f"  {f:40s} {v:.2f}")
    if a.save:
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "extraction_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nsaved {OUT_DIR / 'extraction_eval.json'}")


if __name__ == "__main__":
    main()
