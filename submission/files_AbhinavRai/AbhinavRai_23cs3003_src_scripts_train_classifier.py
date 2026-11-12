"""Trains the TF-IDF + Logistic Regression model on the synthetic corpus (all templates, with OCR noise)
and saves it where the API loads it.

    python scripts/train_classifier.py [--per-template 60] [--noise 0.06]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.classification import ml_classifier as ml
from app.classification.synthetic import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    # clean + noisy copies: the model must cope with real (imperfect) OCR output
    data = generate(a.per_template, noise=0.0, seed=a.seed) + generate(a.per_template, noise=a.noise, seed=a.seed + 1)
    model = ml.train([s.text for s in data], [s.label for s in data])
    ml.save(model, {"docs": len(data), "noise": a.noise, "source": "synthetic corpus (app/classification/synthetic.py)"})
    print(f"trained on {len(data)} documents, saved to {ml.MODEL_PATH}")


if __name__ == "__main__":
    main()
