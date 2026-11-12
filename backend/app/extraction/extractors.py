"""Category-specific extraction rules (regular expressions + label/value parsing).

Every extractor returns one Field per schema entry; a field that is not found stays None
(documents are never assumed to be complete). Labels are listed in priority order, so the
most specific label wins. Label flags: '^x' must start the line, 'x!' needs ':' or '-'.
"""
import re
from collections.abc import Callable

from app.extraction import engine
from app.extraction.engine import Candidate, Field, Line, LabelHit, build_lines, finalize, find_labeled, following_lines
from app.extraction.normalizers import (
    clean_name, clean_text, find_amounts, find_dates, find_id, parse_amount,
)
from app.extraction.schema import specs

# ---- labels ----------------------------------------------------------------------------------
L = {
    "policy_number": ["policy number", "policy no", "policy #", "policy id", "certificate number", "certificate no"],
    "policy_holder": ["policyholder", "policy holder", "insured name", "name of insured", "name of the insured", "life assured",
                      "proposer name", "customer name", "^insured!", "^insured", "^name!"],
    "company": ["insurer!", "insurance company", "issued by", "underwritten by", "company name"],
    "period": ["period of insurance", "policy period", "period of cover", "validity period", "insurance period"],
    "start": ["policy start date", "start date", "inception date", "date of commencement", "commencement date",
              "policy commencement date", "effective date", "valid from", "risk start date"],
    "end": ["policy end date", "policy expiry date", "end date", "expiry date", "date of expiry", "valid until",
            "valid upto", "valid till", "expires on", "date of expiration", "expiry!"],
    "premium": ["total premium payable", "premium payable", "annual premium for renewal", "annual premium", "total premium",
                "premium amount", "premium paid", "renewal premium", "gross premium", "net premium", "premium"],
    "coverage": ["sum insured", "sum assured", "insured declared value", "coverage amount", "coverage limit", "sum covered",
                 "cover amount", "idv", "coverage"],
    "claim_number": ["claim number", "claim no", "claim reference", "claim ref", "claim id", "reference number", "^reference!"],
    "claimant": ["claimant name", "name of claimant", "^claimant", "insured member", "reporting party", "insured name",
                 "policyholder", "policy holder", "customer name", "^insured!", "^insured", "^name!"],
    "incident_date": ["date of incident", "incident date", "date of loss", "date and time of loss", "accident date",
                      "date of accident", "date of occurrence", "loss date", "when!"],
    "claim_amount": ["claim amount", "amount claimed", "total amount claimed", "total claimed", "claimed amount", "sum claimed",
                     "amount of claim", "claim value", "value of the item claimed", "estimated loss", "loss report estimate",
                     "estimated cost of repair"],
    "description": ["description of incident", "description of loss", "describe how the accident happened",
                    "circumstances of the accident", "what happened", "cause of loss", "nature of claim",
                    "details of the claim", "brief description", "incident description"],
    "hospital": ["hospital name", "name of hospital"],
    "patient": ["patient name", "name of patient", "^patient!", "^patient"],
    "bill_number": ["invoice number", "invoice no", "bill number", "bill no", "receipt number", "receipt no", "invoice id",
                    "bill id", "inv no", "estimate no"],
    "bill_date": ["invoice date", "bill date", "date of invoice", "date of bill", "receipt date", "date of service",
                  "service date", "repair date", "^date!", "^dated"],
    "med_fallback_date": ["date of discharge", "discharge date", "discharged", "discharge"],
    "total": ["net payable", "grand total", "total amount due", "total bill amount", "total amount", "total payable",
              "total repair cost", "invoice total", "estimate total", "medical expenses total", "amount payable",
              "total received", "net amount", "final amount", "^payable", "^total"],
    "garage": ["garage name", "workshop name", "service centre name", "name of garage", "garage!", "workshop!"],
    "plate": ["vehicle number", "vehicle no", "vehicle reg", "registration number", "reg no", "^reg!", "plate"],
    "model": ["make and model", "make & model", "vehicle model", "model!", "^vehicle!", "^car!", "^bike!"],
    "dob": ["date of birth", "d.o.b", "dob", "birth date"],
    "idexpiry": ["valid upto", "valid till", "valid until", "valid through", "date of expiry", "expiry date", "expires on"],
    "father": ["father's name", "fathers name", "father name", "s/d/w of", "s/o", "d/o", "w/o", "son of", "daughter of"],
    "id_name": ["elector's name", "name of the holder", "name of holder", "^name!", "^name", "^to!"],
    "surname": ["surname"], "given": ["given names", "given name"],
    "id_generic": ["id number", "document number", "card number", "national id no", "identity number"],
}


def _all_labels() -> list[str]:
    return [engine.parse_label(s)[0] for group in L.values() for s in group]


engine.register_stop_labels(_all_labels())

PLATE_RX = re.compile(r"\b([A-Z]{2})\s?(\d{1,2})\s?([A-Z]{1,3})\s?(\d{4})\b")
COMPANY_WORDS = re.compile(r"\b(insurance|assurance|insurers?|general|mutual|life)\b", re.I)
COMPANY_SKIP = re.compile(r"\b(schedule|certificate|policy|renewal|notice|document|proposal|plan|cover|dear|claim)\b", re.I)
HOSP_WORDS = re.compile(r"\b(hospital|clinic|medical cent(?:re|er)|diagnostics?|institute|nursing home|healthcare|pathology|laborator(?:y|ies)|medicals)\b", re.I)
GARAGE_WORDS = re.compile(r"\b(motors|garage|auto works|autocare|service (?:station|cent(?:re|er))|workshop|body shop|automobiles|tyres|collision repair|repair cent(?:re|er)|car care)\b", re.I)
NAME_STOPWORDS = {"declared", "value", "sum", "amount", "premium", "member", "name", "number", "no"}
DOC_WORDS = re.compile(r"\b(invoice|estimate|bill|receipt|tax|final)\b", re.I)


# ---- helpers ---------------------------------------------------------------------------------
def _method(hit: LabelHit) -> str:
    return "label_next_line" if hit.next_line else "label"


def get(lines: list[Line], labels: list[str], kind: str, *, first_only: bool = False) -> Candidate | None:
    """First label hit (in priority order) whose value is valid for `kind`."""
    for hit in find_labeled(lines, labels):
        m = _method(hit)
        page, rest = hit.line.page, hit.rest
        if kind == "date":
            dates = find_dates(rest)
            if dates:
                d = dates[0]
                return Candidate(d.iso or d.raw, d.raw, page, m, corrected=d.corrected and d.iso is not None,
                                 invalid=d.iso is None, label=hit.label)
        elif kind == "amount":
            a = parse_amount(rest)
            if a:
                return Candidate(a.value, a.raw, page, m, corrected=a.corrected, label=hit.label)
        elif kind == "id":
            tok = find_id(rest, first_only)
            if tok:
                return Candidate(tok, tok, page, m, label=hit.label)
        elif kind == "name":
            name = clean_name(rest)
            if name and name.split()[0].lower() not in NAME_STOPWORDS:
                return Candidate(name, re.split(r"\s{2,}|[,;:(]", rest.strip())[0].strip(), page, m, label=hit.label)
        elif kind == "text":
            rest = re.sub(r"^\([^)]*\)\s*[:\-]?\s*", "", rest)  # "(as far as known): text" -> "text"
            t = clean_text(re.split(r"\s{3,}", rest)[0])
            if t:
                return Candidate(t, t, page, m, label=hit.label)
    return None


def _regex(lines: list[Line], pattern: str, method: str = "pattern", flags: int = 0) -> Candidate | None:
    rx = re.compile(pattern, flags)
    for ln in lines:
        m = rx.search(ln.text)
        if m:
            tok = (m.group(1) if m.groups() else m.group(0)).strip()
            return Candidate(tok.upper() if method == "pattern" and tok.isascii() and " " not in tok else tok, tok, ln.page, method)
    return None


def _date_range(lines: list[Line]) -> tuple[Candidate | None, Candidate | None]:
    """'Period of Insurance: 01/04/2025 to 31/03/2026' -> (start, end)."""
    for hit in find_labeled(lines, L["period"]):
        dates = find_dates(hit.rest)
        if len(dates) < 2:  # the second date may be on the next line
            i = next((k for k, l in enumerate(lines) if l is hit.line), None)
            if i is not None and i + 1 < len(lines):
                dates += find_dates(lines[i + 1].text)
        if len(dates) >= 2:
            page = hit.line.page
            mk = lambda d: Candidate(d.iso or d.raw, d.raw, page, "range", corrected=d.corrected and d.iso is not None, invalid=d.iso is None)
            return mk(dates[0]), mk(dates[1])
    return None, None


def _largest_amount(lines: list[Line]) -> Candidate | None:
    best, page = None, 1
    for ln in lines:
        for a in find_amounts(ln.text):
            if not (a.has_currency or "," in a.raw or "." in a.raw):
                continue  # a bare integer is more likely an id (UHID, invoice no.) than money
            if best is None or float(a.value) > float(best.value):
                best, page = a, ln.page
    return Candidate(best.value, best.raw, page, "guess", note="largest amount in the document") if best else None


def _first_date(lines: list[Line]) -> Candidate | None:
    for ln in lines:
        d = next(iter(find_dates(ln.text)), None)
        if d:
            return Candidate(d.iso or d.raw, d.raw, ln.page, "guess", invalid=d.iso is None, note="first date in the document")
    return None


def _top_line(lines: list[Line], keep: Callable[[str], bool], clean: Callable[[str], str | None], n: int = 8) -> Candidate | None:
    for ln in [l for l in lines if l.page == (lines[0].page if lines else 1)][:n]:
        text = clean(ln.text)
        if text and keep(text):
            return Candidate(text, text, ln.page, "heuristic")
    return None


def _segment(line: str, words: re.Pattern, skip: re.Pattern | None = None) -> str | None:
    """Pick the part of 'Name - Doc type' that carries the organisation words."""
    for seg in re.split(r"\s[-–]\s", line):
        seg = seg.strip(" :-")
        if words.search(seg) and not DOC_WORDS.search(seg):
            return seg.title() if seg.isupper() else seg
    return None


# ---- policy ------------------------------------------------------------------------------------
def extract_policy(lines: list[Line]) -> dict[str, Candidate | None]:
    c: dict[str, Candidate | None] = {}
    c["policy_number"] = get(lines, L["policy_number"], "id") or _regex(lines, r"\b(POL[-/ ]?[A-Z0-9][A-Z0-9/-]{4,})", "pattern")
    name = get(lines, L["policy_holder"], "name")
    if not name:
        m = _regex(lines, r"^\s*Dear\s+(?:(?:Mr|Mrs|Ms|Dr)\.?\s+)?([A-Za-z][A-Za-z .'-]{2,40}?)\s*,", "heuristic")
        name = Candidate(m.value.title() if m.value.isupper() else m.value, m.raw, m.page, "heuristic") if m else None
    c["customer_name"] = name

    company = get(lines, L["company"], "text")
    if not company:
        company = _top_line(
            lines, lambda t: bool(COMPANY_WORDS.search(t)) and len(t.split()) <= 7 and ":" not in t,
            lambda t: None if COMPANY_SKIP.search(t) else (re.split(r"\s[-–]\s|\s{3,}", t.strip())[0].strip(" .,") or None), n=6,
        )
    c["insurance_company"] = company

    start, end = get(lines, L["start"], "date"), get(lines, L["end"], "date")
    r_start, r_end = _date_range(lines)
    c["policy_start_date"] = start or r_start
    c["policy_end_date"] = end or r_end
    c["premium_amount"] = get(lines, L["premium"], "amount")
    c["coverage_amount"] = get(lines, L["coverage"], "amount")
    return c


# ---- claim ---------------------------------------------------------------------------------------
def extract_claim(lines: list[Line]) -> dict[str, Candidate | None]:
    c: dict[str, Candidate | None] = {}
    c["claim_number"] = get(lines, L["claim_number"], "id") or _regex(lines, r"\b(CLM[-/ ]?[A-Z0-9][A-Z0-9/-]{3,})", "pattern")
    c["policy_number"] = get(lines, L["policy_number"], "id") or _regex(lines, r"\b(POL[-/ ]?[A-Z0-9][A-Z0-9/-]{4,})", "pattern")
    c["customer_name"] = get(lines, L["claimant"], "name")
    c["incident_date"] = get(lines, L["incident_date"], "date")
    c["claim_amount"] = get(lines, L["claim_amount"], "amount")
    desc = None
    for hit in find_labeled(lines, L["description"]):
        text = clean_text(re.sub(r"^\([^)]*\)\s*[:\-]?\s*", "", following_lines(lines, hit)))
        if text:
            desc = Candidate(text, clean_text(hit.rest) or text, hit.line.page, _method(hit), label=hit.label)
            break
    c["claim_description"] = desc
    return c


# ---- medical bill ------------------------------------------------------------------------------------
def extract_medical(lines: list[Line]) -> dict[str, Candidate | None]:
    c: dict[str, Candidate | None] = {}
    c["hospital_name"] = get(lines, L["hospital"], "text") or _top_line(
        lines, lambda t: bool(HOSP_WORDS.search(t)), lambda t: _segment(t, HOSP_WORDS), n=6
    )
    c["patient_name"] = get(lines, L["patient"], "name")
    c["invoice_number"] = get(lines, L["bill_number"], "id") or get(lines, ["^invoice", "^bill"], "id", first_only=True)
    date = get(lines, L["bill_date"], "date")
    if not date:
        d = get(lines, L["med_fallback_date"], "date")
        date = (
            Candidate(d.value, d.raw, d.page, "heuristic", note="used the discharge date", corrected=d.corrected, invalid=d.invalid)
            if d else _first_date(lines)
        )
    c["invoice_date"] = date
    c["total_amount"] = get(lines, L["total"], "amount") or _largest_amount(lines)
    return c


# ---- repair invoice ---------------------------------------------------------------------------------------
def _plate(lines: list[Line]) -> Candidate | None:
    hit = get(lines, L["plate"], "text")
    for ln in ([hit_line for hit_line in lines if hit and hit.raw in hit_line.text] if hit else []) + lines:
        m = PLATE_RX.search(ln.text.upper())
        if m:
            tok = "".join(m.groups())
            return Candidate(tok, m.group(0), ln.page, "label" if hit else "pattern")
    if hit:  # old-style or unusual plate: trust the labelled text
        tok = re.sub(r"\s+", "", hit.value.upper())
        if 5 <= len(tok) <= 14 and any(ch.isdigit() for ch in tok):
            return Candidate(tok, hit.raw, hit.page, "label")
    return None


def extract_repair(lines: list[Line]) -> dict[str, Candidate | None]:
    c: dict[str, Candidate | None] = {}
    garage = get(lines, L["garage"], "text") or _top_line(
        lines, lambda t: True, lambda t: _segment(t, GARAGE_WORDS), n=6
    ) or _regex(lines, r"(?i)thank you for visiting\s+([A-Za-z][\w &.'-]{1,40})", "heuristic")
    c["garage_name"] = garage

    plate = _plate(lines)
    model_hit = get(lines, L["model"], "text")
    model = None
    if model_hit:
        # drop a printed plate from the model text; keep the original capitalisation ("Hyundai i20", "XUV300")
        model = re.sub(r"\(?\s*" + PLATE_RX.pattern + r"\s*\)?", "", model_hit.value, flags=re.I).strip(" ,()-")
        model = clean_text(re.sub(r"\bkm\b.*$", "", re.split(r"\s{2,}", model)[0], flags=re.I)) if model else None
        if model and model.isupper():
            model = model.title()
    if plate or model:
        value = ", ".join(x for x in (model, plate.value if plate else None) if x)
        base = model_hit if model else plate
        strengths = [x.method for x in (model_hit if model else None, plate) if x]
        method = "label" if "label" in strengths else ("pattern" if "pattern" in strengths else base.method)
        c["vehicle_details"] = Candidate(value, (model_hit.raw if model_hit and model else plate.raw), (base.page), method)
    else:
        c["vehicle_details"] = None

    c["invoice_number"] = get(lines, L["bill_number"], "id") or get(lines, ["^invoice"], "id", first_only=True)
    c["repair_date"] = get(lines, L["bill_date"], "date") or _first_date(lines)
    c["total_repair_cost"] = get(lines, L["total"], "amount") or _largest_amount(lines)
    return c


# ---- identity document ---------------------------------------------------------------------------------------------
ID_TYPES = [
    ("PAN Card", re.compile(r"permanent account number|\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
    ("Aadhaar Card", re.compile(r"aadhaar|unique identification|(?<!\d{4} )\b\d{4} \d{4} \d{4}\b(?! \d{4}\b)")),
    ("Driving Licence", re.compile(r"driving\s+licen[cs]e", re.I)),
    ("Passport", re.compile(r"\bpassport\b", re.I)),
    ("Voter ID", re.compile(r"election commission|elector|\bepic\b", re.I)),
    ("National ID Card", re.compile(r"national identity|identity card", re.I)),
]
ID_PATTERNS = {
    "PAN Card": r"\b([A-Z]{5}\d{4}[A-Z])\b",
    "Aadhaar Card": r"(?<!\d{4} )\b(\d{4} \d{4} \d{4})\b(?! \d{4}\b)",
    "Driving Licence": r"\b([A-Z]{2}\d{2}\s?\d{11})\b",
    "Passport": r"\b([A-Z]\d{7})\b",
    "Voter ID": r"\b([A-Z]{3}\d{7})\b",
}


_TO_LETTER = str.maketrans("0158285", "OISBZSS"[:7])  # digit -> look-alike letter (0 O, 1 I, 5 S, 8 B, 2 Z)
_TO_DIGIT = str.maketrans("OoIlSBZ", "0011582")  # letter -> look-alike digit


def repair_pan(token: str) -> str | None:
    """PAN is AAAAA9999A. Fix look-alike characters by position; None if it cannot match."""
    t = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    if len(t) != 10:
        return None
    fixed = t[:5].translate(_TO_LETTER) + t[5:9].translate(_TO_DIGIT) + t[9].translate(_TO_LETTER)
    return fixed if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", fixed) else None


def _pan_fallback(lines: list[Line]) -> Candidate | None:
    """The PAN is printed under 'Permanent Account Number'. OCR may garble it: repair what can be
    repaired, otherwise return the raw token flagged as not matching the format."""
    for hit in find_labeled(lines, ["permanent account number"]):
        tok = re.sub(r"\s+", "", hit.rest)
        if not 8 <= len(tok) <= 12:
            continue
        fixed = repair_pan(tok)
        if fixed:
            return Candidate(fixed, tok, hit.line.page, "pattern", corrected=True)
        return Candidate(tok.upper(), tok, hit.line.page, "heuristic", invalid=True, note="does not match the PAN format: please check")
    return None


def extract_identity(lines: list[Line]) -> dict[str, Candidate | None]:
    c: dict[str, Candidate | None] = {}
    full = "\n".join(l.text for l in lines)
    doc_type = next((t for t, rx in ID_TYPES if rx.search(full)), None)
    if doc_type:
        m = dict(ID_TYPES)[doc_type].search(full)
        c["document_type"] = Candidate(doc_type, m.group(0), lines[0].page if lines else 1, "pattern")
    else:
        c["document_type"] = None

    idn = _regex(lines, ID_PATTERNS[doc_type]) if doc_type in ID_PATTERNS else None
    if idn is None and doc_type == "PAN Card":
        idn = _pan_fallback(lines)
    if idn is None:
        idn = get(lines, L["id_generic"], "id") or (get(lines, ["passport no", "passport number"], "id") if doc_type == "Passport" else None)
        if idn is None and doc_type == "Driving Licence":
            idn = get(lines, ["dl no", "licence no", "license no"], "id")
    if idn and doc_type == "Aadhaar Card":
        idn = Candidate(idn.value, idn.raw, idn.page, idn.method)
    c["id_number"] = idn

    name = get(lines, L["id_name"], "name")
    sur, giv = get(lines, L["surname"], "name"), get(lines, L["given"], "name")
    if sur and giv:
        name = Candidate(f"{giv.value} {sur.value}", f"{giv.raw} {sur.raw}", sur.page, "label")
    if not name:  # Aadhaar-style cards print the name on the line above the date of birth
        for i, ln in enumerate(lines):
            if i and re.search(r"\b(dob|date of birth)\b", ln.text, re.I):
                n = clean_name(lines[i - 1].text)
                if n and not re.search(r"government|authority|india|identity", lines[i - 1].text, re.I):
                    name = Candidate(n, lines[i - 1].text.strip(), ln.page, "heuristic")
                break
    c["name"] = name
    c["father_name"] = get(lines, L["father"], "name")
    c["date_of_birth"] = get(lines, L["dob"], "date")
    c["expiry_date"] = get(lines, L["idexpiry"], "date")
    return c


EXTRACTORS: dict[str, Callable[[list[Line]], dict[str, Candidate | None]]] = {
    "insurance_policy": extract_policy,
    "claim_form": extract_claim,
    "medical_bill": extract_medical,
    "repair_invoice": extract_repair,
    "identity_document": extract_identity,
}


def extract(category: str | None, pages: list[dict]) -> list[Field]:
    """pages: [{"page", "text", "words", "avg_confidence"}] -> one Field per schema entry."""
    for p in pages:  # normalise typographic quotes so "Father’s Name" matches
        p["text"] = (p.get("text") or "").replace("’", "'").replace("‘", "'").replace(" ", " ")
    fn = EXTRACTORS.get(category or "")
    if fn is None:
        return []
    lines = build_lines(pages)
    found = fn(lines) if lines else {}
    return [finalize(s.name, found.get(s.name), pages) for s in specs(category)]
