"""Building blocks for the extractors: line model, label->value finder, word location,
and confidence scoring.

Confidence = how the value was found (label on the same line is stronger than a guess)
x how well OCR read the words it came from (mean word confidence). A repaired OCR
mistake (e.g. '3l/03/2026' -> '31/03/2026') or an impossible date lowers it further.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz

STRENGTH = {"label": 0.92, "label_next_line": 0.78, "range": 0.9, "pattern": 0.85, "heuristic": 0.7, "guess": 0.5}
SEPARATORS = r"[\s:\-–=.#]*"


@dataclass
class Line:
    page: int
    idx: int
    text: str


def build_lines(pages: list[dict]) -> list[Line]:
    """pages: [{"page": n, "text": str, "words": [...]}] -> flat list of non-empty lines."""
    lines: list[Line] = []
    for p in pages:
        for i, t in enumerate((p.get("text") or "").splitlines()):
            if t.strip():
                lines.append(Line(p["page"], i, t.rstrip()))
    return lines


@dataclass
class Candidate:
    value: str
    raw: str  # the exact snippet in the OCR text (used to find its position on the page)
    page: int
    method: str  # label | label_next_line | range | pattern | heuristic | guess
    note: str | None = None
    corrected: bool = False
    invalid: bool = False
    label: str | None = None


@dataclass
class Field:
    name: str
    value: str | None = None
    confidence: float = 0.0
    page: int | None = None
    bbox: list[float] | None = None
    method: str | None = None
    source: str | None = None
    note: str | None = None


# ---- labels ------------------------------------------------------------------------
@lru_cache(maxsize=None)
def label_regex(label: str) -> re.Pattern:
    parts = [re.escape(p) for p in label.lower().split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s.]*".join(parts) + r"(?![a-z0-9])")


STOP_LABELS: list[str] = []  # filled by extractors.py (all multi-word labels of every field)
_stop_rx: re.Pattern | None = None


def register_stop_labels(labels: list[str]) -> None:
    global _stop_rx
    STOP_LABELS[:] = sorted({lab.lower() for lab in labels if len(lab) >= 8}, key=len, reverse=True)
    _stop_rx = re.compile("|".join(label_regex(lab).pattern for lab in STOP_LABELS)) if STOP_LABELS else None


def _cut_at_stop(rest: str, own: str) -> str:
    """Cut the remainder where the next labelled field starts ("Policy No: X  Holder: Y")."""
    if not _stop_rx:
        return rest
    low = rest.lower()
    for m in _stop_rx.finditer(low):
        if m.start() > 0 and low[m.start():m.end()] != own:
            return rest[: m.start()]
    return rest


def _starts_with_label(text: str) -> bool:
    return bool(_stop_rx and _stop_rx.match(text.lower().lstrip()))


@dataclass
class LabelHit:
    rest: str  # text after the label (possibly from the next line)
    line: Line
    label: str
    next_line: bool = False


def parse_label(spec: str) -> tuple[str, bool, bool]:
    """'^name!' -> ('name', at_line_start=True, needs_colon=True)."""
    start, colon = spec.startswith("^"), spec.endswith("!")
    return spec.strip("^!"), start, colon


def find_labeled(lines: list[Line], labels: list[str], next_line: bool = True) -> list[LabelHit]:
    """All places where one of `labels` appears, in label-priority then reading order.
    The value is whatever follows the label on the same line, or the next line if empty.
    Label syntax: '^label' = must start the line, 'label!' = must be followed by ':' or '-'
    (use these for short generic words like 'name' or 'date')."""
    hits: list[LabelHit] = []
    for spec in labels:
        label, at_start, needs_colon = parse_label(spec)
        rx = label_regex(label)
        for i, ln in enumerate(lines):
            m = rx.search(ln.text.lower())
            if m:
                if at_start and ln.text[: m.start()].strip(" 	•*-.:0123456789)"):
                    continue
                if needs_colon and not re.match(r"\s*[:\-–=]", ln.text[m.end():]):
                    continue
                rest = ln.text[m.end():]
                rest = re.sub(rf"^{SEPARATORS}", "", rest)
                rest = _cut_at_stop(rest, label.lower()).strip()
                if rest:
                    hits.append(LabelHit(rest, ln, label))
                elif next_line and i + 1 < len(lines) and lines[i + 1].page == ln.page:
                    nxt = lines[i + 1].text.strip()
                    if nxt and not _starts_with_label(nxt):
                        hits.append(LabelHit(_cut_at_stop(nxt, label.lower()).strip(), lines[i + 1], label, True))
                continue
            # tolerate OCR damage inside long labels: "Perlod of lnsurance: ..."
            if len(label) >= 10 and not at_start and ":" in ln.text[: len(label) + 12]:
                head, _, tail = ln.text.partition(":")
                if fuzz.ratio(label.lower(), head.lower().strip()) >= 88 and tail.strip():
                    hits.append(LabelHit(_cut_at_stop(tail.strip(), label.lower()).strip(), ln, label))
    return hits or _fuzzy_hits(lines, labels)


def _fuzzy_hits(lines: list[Line], labels: list[str]) -> list[LabelHit]:
    """Second chance when no label matched exactly: tolerate OCR typos inside the label itself
    ("Tota! Amount", "Pollcy Number"). Only long labels are tried, with a high similarity bar."""
    hits: list[LabelHit] = []
    for spec in labels:
        label, at_start, needs_colon = parse_label(spec)
        if len(label) < 12:  # shorter labels differ from other words by one letter ("invoice no" / "invoice to(tal)")
            continue
        for i, ln in enumerate(lines):
            low = ln.text.lower()
            aln = fuzz.partial_ratio_alignment(label, low)
            if aln is None or aln.score < 90 or aln.dest_end - aln.dest_start < len(label) - 2:
                continue
            if at_start and low[: aln.dest_start].strip(" \t•*-.:0123456789)"):
                continue
            tail = ln.text[aln.dest_end:]
            if needs_colon and not re.match(r"\s*[:\-–=]", tail):
                continue
            rest = _cut_at_stop(re.sub(rf"^{SEPARATORS}", "", tail), label).strip()
            if rest:
                hits.append(LabelHit(rest, ln, label))
    return hits


def following_lines(lines: list[Line], hit: LabelHit, max_lines: int = 3) -> str:
    """Continuation lines of a multi-line value (descriptions), stopping at the next label or blank."""
    start = next((i for i, l in enumerate(lines) if l is hit.line), None)
    if start is None:
        return hit.rest
    parts = [hit.rest]
    for l in lines[start + 1 : start + 1 + max_lines]:
        if l.page != hit.line.page or _starts_with_label(l.text) or re.match(r"^[\w ./'()-]{2,30}\s*:", l.text):
            break
        if not l.text.strip()[:1].islower():
            break  # a new sentence (e.g. "I hereby declare...") is not a wrapped continuation
        parts.append(l.text.strip())
    return " ".join(p for p in parts if p)


# ---- locating a value on the page ------------------------------------------------------
def _norm(tok: str) -> str:
    return re.sub(r"[^\w/\-]", "", tok.lower())


def locate(words: list[dict], raw: str) -> tuple[list[float] | None, float | None]:
    """Find the words that make up `raw` on the page. Returns (bbox [x,y,w,h] fractions, mean confidence)."""
    toks = [t for t in (_norm(t) for t in raw.split()) if t]
    if not toks or not words:
        return None, None
    norm_words = [_norm(w["text"]) for w in words]
    for n in range(min(len(toks), 8), 0, -1):  # longest prefix of the value that appears contiguously
        if n < len(toks) and n < 2 and len(toks[0]) < 3:
            break
        seq = toks[:n]
        for i in range(len(words) - n + 1):
            if norm_words[i : i + n] == seq:
                chunk = words[i : i + n]
                x0, y0 = min(w["x"] for w in chunk), min(w["y"] for w in chunk)
                x1, y1 = max(w["x"] + w["w"] for w in chunk), max(w["y"] + w["h"] for w in chunk)
                conf = sum(w["conf"] for w in chunk) / n
                return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)], conf
    return None, None


def finalize(name: str, cand: Candidate | None, pages: list[dict]) -> Field:
    if cand is None:
        return Field(name)
    page = next((p for p in pages if p["page"] == cand.page), None) or {}
    bbox, word_conf = locate(page.get("words") or [], cand.raw)
    ocr = (word_conf if word_conf is not None else page.get("avg_confidence") or 100.0) / 100.0
    ocr = min(max(ocr, 0.3), 1.0)
    conf = STRENGTH[cand.method] * ocr
    if cand.corrected:
        conf *= 0.85
    if cand.invalid:
        conf *= 0.5
    return Field(
        name=name, value=cand.value, confidence=round(conf, 3), page=cand.page, bbox=bbox,
        method=cand.method, source=cand.raw[:120],
        note=cand.note or ("OCR characters were corrected" if cand.corrected else ("not a valid date" if cand.invalid else None)),
    )
