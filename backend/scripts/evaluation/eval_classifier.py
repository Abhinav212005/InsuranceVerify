"""Compares keyword, ML and hybrid classification on a template-held-out synthetic corpus.

    python scripts/evaluation/eval_classifier.py [--per-template 40] [--noise 0.06] [--save]

Train on templates A-C, test on the UNSEEN template D (so the model cannot memorize layouts).
The keyword rules were written by the project authors, so this is an in-house benchmark;
real-document results belong in the final report once public samples are added.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from app.classification import ml_classifier as ml
from app.classification.classifier import classify
from app.classification.keywords import CATEGORIES
from app.classification.synthetic import generate

OUT_DIR = Path(__file__).resolve().parents[3] / "evaluation_results"


def evaluate(per_template: int, noise: float, seed: int) -> dict:
    train = generate(per_template, ["A", "B", "C"], noise=noise, seed=seed)
    test_clean = generate(per_template, ["D"], noise=0.0, seed=seed + 1)
    test_noisy = generate(per_template, ["D"], noise=noise, seed=seed + 2)
    model = ml.train([s.text for s in train], [s.label for s in train])

    results = {}
    for tname, test in (("clean", test_clean), ("noisy", test_noisy)):
        y = [s.label for s in test]
        for method in ("keyword", "ml", "hybrid"):
            pred = [classify(s.text, method, model)["category"] for s in test]
            p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CATEGORIES, average="macro", zero_division=0)
            results[f"{method}/{tname}"] = {
                "accuracy": round(accuracy_score(y, pred), 4), "precision": round(p, 4),
                "recall": round(r, 4), "f1": round(f, 4),
                "confusion": confusion_matrix(y, pred, labels=CATEGORIES).tolist(),
                "per_class": classification_report(y, pred, labels=CATEGORIES, output_dict=True, zero_division=0),
            }
    return {"train_docs": len(train), "test_docs": len(test_clean), "noise": noise, "labels": CATEGORIES, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.06, help="max OCR character error rate in train and noisy test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="write evaluation_results/classifier_eval.json")
    a = ap.parse_args()
    out = evaluate(a.per_template, a.noise, a.seed)

    print(f"train docs: {out['train_docs']} (templates A-C)   test docs: {out['test_docs']} (unseen template D)   noise<={a.noise}")
    print(f"{'method/test':16s}{'accuracy':>10s}{'macro-P':>10s}{'macro-R':>10s}{'macro-F1':>10s}")
    for k, v in out["results"].items():
        print(f"{k:16s}{v['accuracy']:10.3f}{v['precision']:10.3f}{v['recall']:10.3f}{v['f1']:10.3f}")
    print("\nconfusion matrix (hybrid, noisy):  rows=true, cols=predicted")
    print("labels:", [c[:8] for c in CATEGORIES])
    print(np.array(out["results"]["hybrid/noisy"]["confusion"]))
    if a.save:
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "classifier_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nsaved {OUT_DIR / 'classifier_eval.json'}")


if __name__ == "__main__":
    main()
