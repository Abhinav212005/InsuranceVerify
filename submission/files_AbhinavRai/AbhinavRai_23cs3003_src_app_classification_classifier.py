"""Single entry point: keyword, ML or hybrid classification."""
from typing import Any

from sklearn.pipeline import Pipeline

from app.classification import keyword_classifier as kw
from app.classification import ml_classifier as ml
from app.classification.keywords import CATEGORIES
from app.core.errors import AppError

METHODS = ("keyword", "ml", "hybrid")
HYBRID_KEYWORD_WEIGHT = 0.5
MAX_CHARS = 30000


def available_methods(model: Pipeline | None = None) -> list[str]:
    return list(METHODS) if (model or ml.load()) else ["keyword"]


def classify(text: str, method: str = "hybrid", model: Pipeline | None = None) -> dict[str, Any]:
    """Returns category, confidence, per-category probabilities, evidence and the method
    actually used. `hybrid` averages keyword and ML probabilities; if no trained model is
    available it quietly degrades to keyword-only (reported in `method`)."""
    if method not in METHODS:
        raise AppError("INVALID_METHOD", f"method must be one of {list(METHODS)}", 422)
    text = (text or "")[:MAX_CHARS]
    if not text.strip():
        return {"category": "other", "confidence": 0.0, "method": "keyword", "probabilities": {c: 0.0 for c in CATEGORIES},
                "evidence": [], "negatives": [], "scores": {}, "note": "no text to classify"}

    k = kw.classify(text)
    model = model or ml.load()
    if method == "keyword" or model is None:
        used = "keyword"
        probs, evidence = k["probabilities"], k["evidence"]
        out = {**k, "method": used}
        if method != "keyword":
            out["note"] = "no trained ML model found; used keyword rules only"
        return out

    mp = ml.probabilities(model, text)
    if method == "ml":
        probs = mp
    else:
        w = HYBRID_KEYWORD_WEIGHT
        probs = {c: w * k["probabilities"].get(c, 0.0) + (1 - w) * mp.get(c, 0.0) for c in CATEGORIES}
    best = max(probs, key=probs.get)
    return {
        "category": best,
        "confidence": round(probs[best], 3),
        "method": method,
        "probabilities": {c: round(p, 3) for c, p in probs.items()},
        "keyword": {"category": k["category"], "confidence": k["confidence"]},
        "ml": {"category": max(mp, key=mp.get), "confidence": round(max(mp.values()), 3)},
        # keyword evidence is the explanation shown to the user, whichever method won
        "evidence": kw.score(text)["evidence"].get(best, [])[:10],
        "negatives": [],
        "scores": k["scores"],
    }
