"""TF-IDF + Logistic Regression text classifier (trained locally with scikit-learn)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.classification.keywords import CATEGORIES

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"
META_PATH = MODEL_DIR / "tfidf_logreg.json"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # 1-2 word n-grams; sublinear tf damps very frequent words; min_df=2 drops one-off typos
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, max_features=30000)),
            ("clf", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")),
        ]
    )


def train(texts: list[str], labels: list[str]) -> Pipeline:
    model = build_pipeline()
    model.fit(texts, labels)
    return model


def save(model: Pipeline, meta: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {**meta, "trained_at": datetime.now(timezone.utc).isoformat(), "labels": list(model.classes_)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


_cache: dict[str, Pipeline] = {}


def load(path: Path = MODEL_PATH) -> Pipeline | None:
    key = str(path)
    if key not in _cache:
        if not path.exists():
            return None
        _cache[key] = joblib.load(path)
    return _cache[key]


def reset_cache() -> None:
    _cache.clear()


def metadata(path: Path = MODEL_PATH) -> dict | None:
    p = path.with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def probabilities(model: Pipeline, text: str) -> dict[str, float]:
    probs = model.predict_proba([text])[0]
    out = {c: 0.0 for c in CATEGORIES}
    out.update({c: float(p) for c, p in zip(model.classes_, probs)})
    return out
