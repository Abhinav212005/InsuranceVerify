"""Rule/keyword classifier: weighted phrase matching with OCR-typo tolerance.

For every category we sum the weights of the phrases found in the text, add regex
pattern hits, subtract negative phrases, then turn the scores into probabilities.
Because "other" has no keywords of its own, it gets a fixed pseudo-score
(OTHER_PRIOR): when no category collects clearly more evidence than that, the
document falls back to "other" - and confidence honestly reflects the weak evidence.
"""
import re
from functools import lru_cache

from rapidfuzz import fuzz

from app.classification.keywords import CATEGORIES, KEYWORDS, NEGATIVE, PATTERNS

OTHER_PRIOR = 4.0  # evidence needed before a real category beats "other"
FUZZY_MIN_LEN = 8  # shorter phrases are never fuzzy-matched (too many false hits)
FUZZY_THRESHOLD = 90
FUZZY_FACTOR = 0.8  # fuzzy hits count a little less than exact ones
TITLE_CHARS = 400  # phrases in the first lines (document titles) get a boost
TITLE_BOOST = 1.3
MAX_CHARS = 30000


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def _phrase_regex(phrase: str) -> re.Pattern:
    body = r"\s+".join(re.escape(p) for p in phrase.split())
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _find(phrase: str, text: str) -> tuple[int, bool, int]:
    """Returns (count, fuzzy, first_position). Exact word-boundary match first,
    then a tolerant match for longer phrases to survive OCR typos."""
    hits = list(_phrase_regex(phrase).finditer(text))
    if hits:
        return len(hits), False, hits[0].start()
    if len(phrase) >= FUZZY_MIN_LEN and fuzz.partial_ratio(phrase, text) >= FUZZY_THRESHOLD:
        return 1, True, text.find(phrase[:4]) if phrase[:4] in text else len(text)
    return 0, False, -1


def score(text: str) -> dict:
    """Returns {"scores": {cat: float}, "evidence": {cat: [...]}, "negatives": {cat: [...]}}."""
    raw = text[:MAX_CHARS]
    norm = normalize(raw)
    scores = {c: 0.0 for c in CATEGORIES if c != "other"}
    evidence = {c: [] for c in scores}
    negatives = {c: [] for c in scores}

    for cat, words in KEYWORDS.items():
        for phrase, weight in words.items():
            count, fuzzy, pos = _find(phrase, norm)
            if not count:
                continue
            w = weight * (FUZZY_FACTOR if fuzzy else 1.0)
            w *= min(1.0 + 0.25 * (count - 1), 1.75)  # repeats help, with a cap
            if weight >= 4 and 0 <= pos < TITLE_CHARS:
                w *= TITLE_BOOST
            scores[cat] += w
            evidence[cat].append({"keyword": phrase, "weight": round(w, 2), "count": count, "fuzzy": fuzzy})

    for cat, pats in PATTERNS.items():
        for pattern, weight, label in pats:
            if re.search(pattern, raw):
                scores[cat] += weight
                evidence[cat].append({"keyword": label, "weight": float(weight), "count": 1, "fuzzy": False})

    for cat, words in NEGATIVE.items():
        for phrase, weight in words.items():
            if _phrase_regex(phrase).search(norm):
                scores[cat] -= weight
                negatives[cat].append(phrase)

    for cat in scores:
        scores[cat] = round(max(scores[cat], 0.0), 2)
        evidence[cat].sort(key=lambda e: -e["weight"])
    return {"scores": scores, "evidence": evidence, "negatives": negatives}


def probabilities(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values()) + OTHER_PRIOR
    probs = {c: s / total for c, s in scores.items()}
    probs["other"] = OTHER_PRIOR / total
    return probs


def classify(text: str) -> dict:
    s = score(text)
    probs = probabilities(s["scores"])
    best = max(probs, key=probs.get)
    if best != "other" and probs[best] <= probs["other"]:
        best = "other"  # a tie is not "clearly more evidence than nothing"
    return {
        "category": best,
        "confidence": round(probs[best], 3),
        "probabilities": {c: round(p, 3) for c, p in probs.items()},
        "scores": s["scores"],
        "evidence": s["evidence"].get(best, [])[:10],
        "negatives": s["negatives"].get(best, []),
    }
