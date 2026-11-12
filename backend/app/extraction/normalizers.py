"""Turn raw OCR snippets into clean values: ISO dates, decimal amounts, names, identifiers.

OCR confuses look-alike characters (O/0, l/1, S/5, B/8). Inside tokens that are
otherwise numeric we repair them and report `corrected=True` so confidence can drop.
"""
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"})
_D = r"[0-9OoIl|SB]"  # a digit, or something OCR commonly mistakes for one
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"


@dataclass
class DateHit:
    raw: str
    iso: str | None  # None when the text looks like a date but is impossible (e.g. 31/02/2025)
    start: int
    end: int
    corrected: bool = False


def _digits(s: str) -> tuple[int, bool]:
    fixed = s.translate(_FIX)
    return int(fixed), fixed != s


def _year(y: int, digits: int) -> int:
    if digits == 2:
        return 2000 + y if y <= 69 else 1900 + y
    return y


def _make(d: int, m: int, y: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


_DATE_PATTERNS = [
    # day first (Indian convention), 4- or 2-digit year
    ("dmy", re.compile(rf"(?<![0-9A-Za-z/.-])({_D}{{1,2}})\s?[/.\-]\s?({_D}{{1,2}})\s?[/.\-]\s?({_D}{{4}}|{_D}{{2}})(?![0-9A-Za-z])")),
    ("ymd", re.compile(rf"(?<![0-9A-Za-z/.-])({_D}{{4}})\s?[/.\-]\s?({_D}{{1,2}})\s?[/.\-]\s?({_D}{{1,2}})(?![0-9A-Za-z])")),
    ("d_mon_y", re.compile(rf"(?<![0-9A-Za-z])(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]+{_MON}[\s,\-]+(\d{{4}})(?![0-9])", re.I)),
    ("mon_d_y", re.compile(rf"(?<![0-9A-Za-z]){_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})(?![0-9])", re.I)),
]


def find_dates(text: str) -> list[DateHit]:
    hits: list[DateHit] = []
    taken: list[tuple[int, int]] = []
    for kind, rx in _DATE_PATTERNS:
        for m in rx.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            g = m.groups()
            corrected = False
            try:
                if kind == "dmy":
                    d, c1 = _digits(g[0]); mo, c2 = _digits(g[1]); y, c3 = _digits(g[2])
                    y = _year(y, len(g[2]))
                    if mo > 12 >= d:  # clearly month-first (US style)
                        d, mo = mo, d
                    corrected = c1 or c2 or c3
                elif kind == "ymd":
                    y, c1 = _digits(g[0]); mo, c2 = _digits(g[1]); d, c3 = _digits(g[2])
                    corrected = c1 or c2 or c3
                elif kind == "d_mon_y":
                    d, y, mo = int(g[0]), int(g[2]), MONTHS[g[1][:3].lower()]
                else:
                    mo, d, y = MONTHS[g[0][:3].lower()], int(g[1]), int(g[2])
            except (ValueError, KeyError):
                continue
            hits.append(DateHit(m.group(0), _make(d, mo, y), m.start(), m.end(), corrected))
            taken.append((m.start(), m.end()))
    return sorted(hits, key=lambda h: h.start)


# --- amounts -----------------------------------------------------------------
# '12,450.00', '5,00,000', and '31.600' (OCR often turns the thousands comma into a period)
_NUM = rf"(?:{_D}{{1,3}}(?:\.{_D}{{3}})+(?![0-9])|{_D}{{1,3}}(?:,{_D}{{2,3}})+|{_D}+)(?:\.{_D}{{1,2}})?"
_AMOUNT_RX = re.compile(
    rf"(?:(?P<cur>(?:rs|inr|usd|eur|rupees?)\.?\s*|₹\s*|\$\s*)|(?<![A-Za-z0-9/.\-,]))(?P<num>{_NUM})(?![0-9A-Za-z/\-]|\.\d|%)",
    re.I,
)


@dataclass
class AmountHit:
    raw: str
    value: str  # "12450.00"
    has_currency: bool
    start: int
    end: int
    corrected: bool = False


def find_amounts(text: str) -> list[AmountHit]:
    out = []
    for m in _AMOUNT_RX.finditer(text):
        num = m.group("num")
        if not any(c.isdigit() for c in num):
            continue  # "OOO" or "lll" is not an amount
        fixed = num.translate(_FIX).replace(",", "")
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", fixed):  # period used as thousands separator
            fixed = fixed.replace(".", "")
        try:
            val = Decimal(fixed)
        except InvalidOperation:
            continue
        # a lone small integer (e.g. "5" in "Table 5") is rarely money unless a currency marker is present
        if not m.group("cur") and val < 10 and "." not in fixed:
            continue
        out.append(AmountHit(m.group(0).strip(), f"{val:.2f}", bool(m.group("cur")), m.start(), m.end(), num.translate(_FIX) != num))
    return out


def parse_amount(text: str) -> AmountHit | None:
    hits = find_amounts(text)
    if not hits:
        return None
    return next((h for h in hits if h.has_currency), hits[0])


# --- names & ids -----------------------------------------------------------------
_SALUTATION = re.compile(r"^(?:mr|mrs|ms|miss|dr|shri|smt|sri|prof|mx)\.?\s+", re.I)


def clean_name(raw: str) -> str | None:
    s = re.split(r"\s{2,}|\t|[,;:|(]|\d", raw.strip())[0]
    s = _SALUTATION.sub("", s.strip(" .-_"))
    s = re.sub(r"\s+", " ", s).strip(" .-_")
    words = s.split()
    if not 1 <= len(words) <= 6 or not 3 <= len(s) <= 60:
        return None
    letters = sum(c.isalpha() for c in s)
    if letters / len(s.replace(" ", "")) < 0.9 or not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]*", s):
        return None
    return s.title() if s.isupper() else s


_ID_RX = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9\-/]{3,34})(?![A-Za-z0-9\-/])")


_DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}")


def find_id(text: str, first_only: bool = False) -> str | None:
    """First token that looks like a reference number: has a digit, 4+ chars, not a date.
    With first_only the id must be the very first word (for loose labels like 'Invoice ...')."""
    for m in _ID_RX.finditer(text):
        tok = m.group(1).strip("-/")
        if first_only and text[: m.start()].strip():
            return None
        if any(c.isdigit() for c in tok) and len(tok) >= 4 and not _DATE_LIKE.fullmatch(tok):
            return tok.upper()
        if first_only:
            return None
    return None


def clean_text(raw: str, limit: int = 300) -> str | None:
    s = re.sub(r"\s+", " ", raw).strip(" :-_|")
    return s[:limit] or None
