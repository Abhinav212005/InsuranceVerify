"""Downloadable report for one document: JSON for machines, PDF for people."""
from datetime import datetime, timezone
from textwrap import wrap

import fitz  # PyMuPDF

from app.db.models import Document
from app.extraction.schema import specs

DISCLAIMER = (
    "This report reflects data completeness and OCR confidence only. It is not an approval or rejection "
    "of any insurance claim. Values marked 'edited' were corrected by a person."
)


def build_report(doc: Document) -> dict:
    labels = {s.name: s.label for s in specs(doc.doc_category)}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": {
            "id": doc.id, "filename": doc.filename, "file_type": doc.file_type, "pages": doc.page_count,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            "processing_ms": doc.processing_ms, "status": doc.status,
        },
        "classification": {
            "category": doc.doc_category, "confidence": doc.category_confidence, "method": doc.classification_method,
        },
        "review_status": doc.review_status,
        "fields": [
            {
                "name": f.field_name, "label": labels.get(f.field_name, f.field_name), "value": f.field_value,
                "confidence": f.confidence, "edited": f.is_edited, "original_value": f.original_value, "page": f.source_page,
            }
            for f in doc.fields
        ],
        "issues": [
            {"code": i.rule_code, "severity": i.severity, "field": i.field_name, "message": i.message} for i in doc.issues
        ],
        "ocr": [
            {"page": p.page_number, "source": p.ocr_source, "average_confidence": p.avg_confidence, "text": p.ocr_text or ""}
            for p in doc.pages
        ],
        "disclaimer": DISCLAIMER,
    }


class _Pdf:
    """Tiny text layout helper on top of PyMuPDF (A4, helvetica, automatic page breaks)."""

    W, H, M = 595, 842, 50

    def __init__(self, title: str):
        self.doc = fitz.open()
        self.title = title
        self.page = None
        self.y = 0.0
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page(width=self.W, height=self.H)
        self.y = self.M
        self.page.insert_text((self.M, self.H - 25), DISCLAIMER[:118], fontsize=6.5, color=(0.4, 0.4, 0.4))
        self.page.insert_text((self.W - self.M - 40, self.H - 25), f"Page {self.doc.page_count}", fontsize=6.5, color=(0.4, 0.4, 0.4))

    def line(self, text: str, size: float = 9.5, bold: bool = False, color=(0, 0, 0), indent: float = 0, gap: float = 3):
        font = "hebo" if bold else "helv"
        width_chars = int((self.W - 2 * self.M - indent) / (size * 0.5))
        for part in wrap(text, width_chars) or [""]:
            if self.y > self.H - 60:
                self._new_page()
            self.page.insert_text((self.M + indent, self.y + size), part, fontname=font, fontsize=size, color=color)
            self.y += size + gap

    def space(self, h: float = 8):
        self.y += h

    def bytes(self) -> bytes:
        return self.doc.tobytes()


def to_pdf(report: dict) -> bytes:
    d = report["document"]
    pdf = _Pdf(d["filename"])
    pdf.line("Document report", 18, bold=True)
    pdf.line(d["filename"], 11, color=(0.25, 0.25, 0.25))
    pdf.space(6)
    status = {"valid": "Valid (data complete)", "needs_review": "Needs review", "incomplete": "Incomplete"}.get(report["review_status"], "Not validated")
    c = report["classification"]
    cat = (c["category"] or "unknown").replace("_", " ").title()
    conf = f"{c['confidence']:.0%}" if c["confidence"] is not None else "n/a"
    pdf.line(f"Review status: {status}", 11, bold=True)
    pdf.line(f"Document type: {cat} (confidence {conf}, method {c['method'] or 'n/a'})")
    pdf.line(f"Pages: {d['pages']}    Uploaded: {d['uploaded_at'] or 'n/a'}    Generated: {report['generated_at']}")
    pdf.space(10)

    pdf.line("Extracted fields", 13, bold=True)
    if not report["fields"]:
        pdf.line("No structured fields for this document type.", color=(0.4, 0.4, 0.4))
    for f in report["fields"]:
        value = f["value"] if f["value"] else "(not found)"
        tag = " [edited]" if f["edited"] else f"  ({f['confidence']:.0%})" if f["value"] else ""
        pdf.line(f"{f['label']}: {value}{tag}", indent=6)
    pdf.space(10)

    pdf.line("Validation", 13, bold=True)
    if not report["issues"]:
        pdf.line("No issues.", color=(0.4, 0.4, 0.4))
    for i in report["issues"]:
        colour = {"error": (0.75, 0.1, 0.1), "warning": (0.7, 0.4, 0), "info": (0.2, 0.4, 0.6)}[i["severity"]]
        pdf.line(f"[{i['severity'].upper()}] {i['message']}", color=colour, indent=6)
    pdf.space(10)

    pdf.line("Recognised text", 13, bold=True)
    for p in report["ocr"]:
        avg = f", confidence {p['average_confidence']:.0f}%" if p["average_confidence"] is not None else ""
        pdf.line(f"Page {p['page']} ({p['source'] or 'not read'}{avg})", 10, bold=True, gap=2)
        for text_line in (p["text"] or "(no text)").splitlines()[:80]:
            pdf.line(text_line, 8.5, indent=6, gap=1.5)
        pdf.space(4)
    pdf.space(6)
    pdf.line(DISCLAIMER, 8, color=(0.4, 0.4, 0.4))
    return pdf.bytes()
