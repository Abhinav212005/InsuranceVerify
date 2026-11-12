"""Builds every submission file.   python build.py "Your Name" 21BCE1234 [--out ../files]"""
import shutil
import sys
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image as RLImage, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)

import content as C
IMG = C.IMG

NAME = sys.argv[1] if len(sys.argv) > 1 else "Your Name"
ROLL = sys.argv[2] if len(sys.argv) > 2 else "RollNumber"
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--out" else Path(__file__).parent.parent / "files"
OUT.mkdir(parents=True, exist_ok=True)
PREFIX = f"{NAME.replace(' ', '')}_{ROLL}_"
ROOT = Path(__file__).resolve().parents[2]
AFFIL = "Digital Image Processing (DIP) Course Project"


def S(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=9.2, leading=10.6, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)


# ------------------------------------------------------------------------------ shared flowable builders
def blocks_to_flow(blocks, st, col_w, big=False):
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "long":
            b, kind = b[1], b[1][0]
        if kind == "p":
            out.append(Paragraph(b[1], st["p"]))
        elif kind == "h":
            out.append(Paragraph(b[1], st["h2"]))
        elif kind == "fig":
            _, path, cap, frac = b
            w = col_w * frac
            iw, ih = Image.open(path).size
            img = RLImage(str(path), width=w, height=w * ih / iw)
            out.append(KeepTogether([img, Spacer(1, 2), Paragraph(cap, st["cap"]), Spacer(1, 4)]))
        elif kind == "table":
            rows, cap = b[1], b[2]
            n = len(rows[0])
            fs = st["tfs"]
            data = [[Paragraph(str(c), ParagraphStyle("tc", fontName="Times-Bold" if r == 0 else "Times-Roman", fontSize=fs,
                                                       leading=fs + 1.2, alignment=1 if i else 0))
                     for i, c in enumerate(row)] for r, row in enumerate(rows)]
            if st["wide_first"] and n > 2:
                cw = [col_w * st["wide_first"]] + [col_w * (1 - st["wide_first"]) / (n - 1)] * (n - 1)
            else:
                cw = None
            t = Table(data, colWidths=cw, hAlign="CENTER")
            t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                                   ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("TOPPADDING", (0, 0), (-1, -1), 1),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            out.append(KeepTogether([Paragraph(cap, st["tcap"]), t, Spacer(1, 5)]))
        elif kind == "code":
            _, text, cap = b
            pre = Preformatted(text, ParagraphStyle("code", fontName="Courier", fontSize=st["codefs"], leading=st["codefs"] + 1.1))
            box = Table([[pre]], colWidths=[col_w])
            box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, -1), colors.Color(.96, .96, .96)),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            out.append(KeepTogether([box, Spacer(1, 2), Paragraph(cap, st["cap"]), Spacer(1, 4)]))
    return out


def ref_flow(st):
    return [Paragraph(f"[{i}] {r}", st["ref"]) for i, r in enumerate(C.REFS, 1)]


# ------------------------------------------------------------------------------ IEEE 4-page paper
def build_ieee(path):
    W, H = letter
    lm = rm = 0.62 * inch
    tm, bm = 0.7 * inch, 0.75 * inch
    gap = 0.22 * inch
    cw = (W - lm - rm - gap) / 2
    full_w = W - lm - rm
    title_h = 3.0 * inch
    fr_title = Frame(lm, H - tm - title_h, full_w, title_h, id="t", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_h = H - tm - bm
    c1 = Frame(lm, bm, cw, H - tm - title_h - bm, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c2 = Frame(lm + cw + gap, bm, cw, H - tm - title_h - bm, id="c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    d1 = Frame(lm, bm, cw, f_h, id="d1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    d2 = Frame(lm + cw + gap, bm, cw, f_h, id="d2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(W / 2, 0.4 * inch, str(doc.page))

    doc = BaseDocTemplate(str(path), pagesize=letter, title=C.TITLE, author=NAME,
                          pageTemplates=[PageTemplate("first", [fr_title, c1, c2], onPage=footer),
                                         PageTemplate("later", [d1, d2], onPage=footer)])
    st = dict(p=S("p"), h2=S("h2", fontName="Times-BoldItalic", fontSize=9.2, alignment=0, spaceBefore=3, spaceAfter=1),
              cap=S("cap", fontSize=7.8, leading=9, alignment=TA_CENTER), tcap=S("tcap", fontSize=7.8, leading=9, alignment=TA_CENTER, spaceAfter=2),
              ref=S("ref", fontSize=7.6, leading=8.6, leftIndent=12, firstLineIndent=-12), tfs=7.4, wide_first=0.4, codefs=5.6)
    H1 = S("h1", fontName="Times-Roman", fontSize=9.2, alignment=TA_CENTER, spaceBefore=5, spaceAfter=2)
    story = [
        Paragraph(C.TITLE, S("title", fontSize=20, leading=22, alignment=TA_CENTER, fontName="Times-Roman")), Spacer(1, 8),
        Paragraph(f"{NAME}<br/>Roll No. {ROLL}<br/>{AFFIL}", S("au", fontSize=10.5, leading=13, alignment=TA_CENTER)), Spacer(1, 8),
        Paragraph(f"<b><i>Abstract</i></b>—<b>{C.ABSTRACT}</b>", S("abs", fontName="Times-Bold", fontSize=8.8, leading=10)), Spacer(1, 3),
        Paragraph(f"<b><i>Index Terms</i></b>—<b>{C.KEYWORDS}</b>", S("kw", fontName="Times-Bold", fontSize=8.8, leading=10)),
        FrameBreak(),
        NextPageTemplate("later"),
    ]
    secs = [("I. PROBLEM STATEMENT", C.PROBLEM), ("II. STATE OF THE ART", C.SOTA), ("III. DATASET", C.DATA),
            ("IV. METHODOLOGY", C.METHOD), ("V. IMPLEMENTATION", C.IMPL), ("VI. NOVELTY AND USP USING DIP", C.NOVELTY),
            ("VII. RESULTS AND DASHBOARD", C.RESULTS), ("VIII. CONCLUSION", C.CONCL)]
    for title, blocks in secs:
        story.append(Paragraph(title, H1))
        story += blocks_to_flow(blocks, st, cw)
    story.append(Paragraph("REFERENCES", H1))
    story += ref_flow(st)
    doc.build(story)


# ------------------------------------------------------------------------------ report-style docs (A4, one column)
def report_styles():
    return dict(p=S("p", fontSize=10.5, leading=14, spaceAfter=5), h2=S("h2", fontName="Times-Bold", fontSize=11.5, leading=14, alignment=0, spaceBefore=8),
                cap=S("cap", fontSize=9, leading=11, alignment=TA_CENTER), tcap=S("tcap", fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=2),
                ref=S("ref", fontSize=10, leading=12.5, leftIndent=18, firstLineIndent=-18, spaceAfter=3), tfs=9, wide_first=None, codefs=8)


def build_report_doc(path, title, sections, cover=True, full=False):
    st = report_styles()
    W, H = A4
    cw = W - 2 * inch

    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(W / 2, 0.55 * inch, f"{doc.page}")
        canvas.drawString(inch, 0.55 * inch, f"{NAME} ({ROLL})")

    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=0.9 * inch, title=title, author=NAME)
    doc.addPageTemplates([PageTemplate("p", [Frame(inch, 0.9 * inch, cw, H - 1.9 * inch, id="f")], onPage=footer)])
    h1 = S("rh1", fontName="Times-Bold", fontSize=15, leading=18, alignment=0, spaceBefore=6, spaceAfter=8)
    story = []
    if cover:
        story += [Spacer(1, 1.6 * inch), Paragraph(C.TITLE, S("ct", fontName="Times-Bold", fontSize=22, leading=27, alignment=TA_CENTER)), Spacer(1, 0.3 * inch),
                  Paragraph(title if title != C.TITLE else "Project Report", S("cs", fontSize=14, leading=18, alignment=TA_CENTER)), Spacer(1, 0.6 * inch),
                  Paragraph(f"<b>{NAME}</b><br/>Roll Number: {ROLL}<br/>{AFFIL}", S("ca", fontSize=12, leading=17, alignment=TA_CENTER)), Spacer(1, 0.5 * inch),
                  Paragraph("This system assists data entry. It does not approve or reject insurance claims. All data is synthetic.",
                            S("cn", fontName="Times-Italic", fontSize=10.5, leading=13, alignment=TA_CENTER))]
        if full:
            story += [PageBreak(), Paragraph("Abstract", h1), Paragraph(C.ABSTRACT, st["p"]), Paragraph(f"<b>Keywords:</b> {C.KEYWORDS}", st["p"])]
            story += blocks_to_flow(C.LONG_EXTRA["intro"], st, cw)
        story.append(PageBreak())
    for i, (sec_title, blocks) in enumerate(sections):
        story.append(Paragraph(sec_title, h1))
        story += blocks_to_flow(blocks, st, cw)
        if i < len(sections) - 1 and full:
            story.append(Spacer(1, 6))
    doc.build(story)


def build_components():
    method = C.METHOD
    comps = {
        "1_ProblemStatement": ("Problem Statement", [("1. Problem Statement", C.PROBLEM)]),
        "2_StateOfTheArt": ("State of the Art", [("2. State of the Art", C.SOTA)]),
        "3_Methodology_Flowchart_Pseudocode": ("Methodology with Flowchart and Pseudocode", [("3. Methodology", method)]),
        "4_Implementation": ("Implementation", [("4. Implementation", C.IMPL + C.LONG_EXTRA["dip"])]),
        "5_Dataset": ("Dataset and Data Source", [("5. Dataset", C.DATA)]),
        "6_Novelty_USP_DIP": ("Novelty and USP using DIP", [("6. Novelty and USP using DIP", C.NOVELTY)]),
        "7_Results_Dashboards": ("Results and Dashboards", [("7. Results and Dashboards", C.RESULTS + C.LONG_EXTRA["eval"])]),
        "8_References": ("References", [("8. References", [])]),
    }
    for fname, (ttl, secs) in comps.items():
        p = OUT / f"{PREFIX}{fname}.pdf"
        if fname == "8_References":
            st = report_styles()
            doc = BaseDocTemplate(str(p), pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch, title=ttl, author=NAME)
            doc.addPageTemplates([PageTemplate("p", [Frame(inch, inch, A4[0] - 2 * inch, A4[1] - 2 * inch)])])
            doc.build([Paragraph(C.TITLE, S("rt", fontName="Times-Bold", fontSize=13, leading=16, alignment=0)),
                       Paragraph(f"References — {NAME} ({ROLL})", S("rs", fontSize=11, alignment=0, spaceAfter=8))] + ref_flow(st))
        else:
            build_report_doc(p, ttl, secs, cover=False)
        print("built", p.name)


def build_full_report(path):
    secs = [("1. Introduction and Problem Statement", C.PROBLEM), ("2. State of the Art", C.SOTA), ("3. Dataset and Data Source", C.DATA),
            ("4. Methodology", C.METHOD + C.LONG_EXTRA["dip"]), ("5. Implementation and Source Code", C.IMPL),
            ("6. Novelty and USP using DIP", C.NOVELTY), ("7. Results and Dashboards", C.RESULTS + C.LONG_EXTRA["eval"]),
            ("8. Conclusion, Limitations and Future Work", C.CONCL)]
    # references appended as a final section through a tiny wrapper
    refs = [("p", f"[{i}] {r}") for i, r in enumerate(C.REFS, 1)]
    secs.append(("9. References", refs))
    build_report_doc(path, C.TITLE, secs, cover=True, full=True)


# ------------------------------------------------------------------------------ PPT
def build_ppt(path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    blank = prs.slide_layouts[6]
    INDIGO, SLATE = RGBColor(0x43, 0x38, 0xCA), RGBColor(0x1E, 0x29, 0x3B)

    def slide(title, bullets=None, img=None, img_w=7.2, note=None, small=False):
        s = prs.slides.add_slide(blank)
        bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12)); bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
        tb = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.9)); tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = SLATE
        wtxt = 5.4 if img and bullets else 12
        if bullets:
            tb = s.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(wtxt), Inches(5.6)); tf = tb.text_frame; tf.word_wrap = True
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "• " + b; p.font.size = Pt(14 if small else 18); p.font.color.rgb = SLATE; p.space_after = Pt(8)
        if img:
            iw, ih = Image.open(img).size
            w = Inches(img_w if bullets else 11.5); h = int(w * ih / iw)
            maxh = Inches(5.6)
            if h > maxh: h = maxh; w = int(h * iw / ih)
            left = Inches(6.2) if bullets else int((prs.slide_width - w) / 2)
            s.shapes.add_picture(str(img), left, Inches(1.4), width=w, height=h)
        ft = s.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12), Inches(0.35)); ft.text_frame.text = f"{NAME} | {ROLL} | synthetic data only; assists data entry, never approves claims"
        ft.text_frame.paragraphs[0].font.size = Pt(10); ft.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
        if note: s.notes_slide.notes_text_frame.text = note
        return s

    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.4)); tf = tb.text_frame; tf.word_wrap = True
    tf.text = C.TITLE; tf.paragraphs[0].font.size = Pt(36); tf.paragraphs[0].font.bold = True; tf.paragraphs[0].font.color.rgb = SLATE
    p = tf.add_paragraph(); p.text = f"{NAME}  |  Roll No. {ROLL}"; p.font.size = Pt(20); p.font.color.rgb = INDIGO
    p = tf.add_paragraph(); p.text = AFFIL; p.font.size = Pt(16)
    slide("1. Problem statement", ["Staff retype policy numbers, dates and amounts from scans, phone photos and PDFs",
                                   "Skew, blur, noise, shadows and low resolution make plain OCR unreliable",
                                   "Goal: improve the image with DIP, extract text, classify, extract fields, validate, keep a human in control",
                                   "Scope: assists data entry; never approves or rejects claims"])
    slide("2. State of the art", ["Binarisation: Otsu (global), Sauvola / adaptive (local)", "Denoising: median keeps edges, Gaussian blurs strokes",
                                  "Skew: projection profile (Postl), Hough; page boundary: Canny + contours + perspective",
                                  "OCR: Tesseract LSTM; deep layout models need large data and GPUs",
                                  "Gap: few intake demos measure what each preprocessing step contributes"])
    slide("3. Methodology: workflow", None, IMG / "flowchart.png")
    slide("3. Methodology: pseudocode", None, None)
    s = prs.slides[-1]
    tb = s.shapes.add_textbox(Inches(0.6), Inches(1.3), Inches(6), Inches(5.6)); tb.text_frame.word_wrap = True
    tb.text_frame.text = C.PSEUDO_PIPELINE
    for p in tb.text_frame.paragraphs: p.font.name = "Consolas"; p.font.size = Pt(10)
    tb = s.shapes.add_textbox(Inches(6.9), Inches(1.3), Inches(6), Inches(5.6)); tb.text_frame.word_wrap = True
    tb.text_frame.text = C.PSEUDO_SKEW + "\n\n" + C.PSEUDO_STATUS
    for p in tb.text_frame.paragraphs: p.font.name = "Consolas"; p.font.size = Pt(10)
    slide("4. Dataset", ["48 synthetic documents: 6 categories x 8, four layouts and fonts, exact ground truth", "15 controlled degradations, 720 images",
                         "360 defect documents for validation; 24 independent hand-written texts as a sanity check",
                         "No real personal data. Public corpora (FUNSD, SROIE, RVL-CDIP) cited, not used"])
    slide("5. DIP in action", None, IMG / "stages.png")
    slide("6. Novelty and USP using DIP", ["Default recipe chosen by ablation, not convention (CLAHE hurts on clean text)", "Explainable deskew: mean error 0.04 degrees",
                                           "Honest negative results (Gaussian blur, low-DPI binarisation)", "Image Processing Studio shows every stage and metric",
                                           "Fail-safe boundary detection; human review with edit tracking"], small=False)
    slide("7. Result: preprocessing ablation", ["CER 15.7% -> 3.2% (-79.8%)", "WER 20.9% -> 11.8%", "270 images, 15 conditions",
                                                 "Adaptive threshold alone hurts WER; deskew and morphology recover it"], CH_IMG("01_ablation_bar.png"), 7.0)
    slide("8. Result: deskew and denoising", ["Angle error 0.04 degrees", "CER at 8 degrees: 0.563 -> 0.016", "Median 5: SSIM 0.967, CER 0.017", "Gaussian 5x5: CER 0.238"],
          CH_IMG("04_skew.png"), 7.0)
    slide("9. Result: classification and extraction", ["Keyword 99.2%, hybrid 99.4% end to end", "ML alone: 72-79% on held-out text", "Fields: 88.6% end to end, 96.6% on independent docs",
                                                       "Validation status correct on 360 defect docs"], CH_IMG("08_confusion.png"), 5.5)
    slide("10. Dashboard", None, IMG / "dashboard.png")
    slide("11. Document details and validation", None, IMG / "detail.png")
    slide("12. Limitations and future work", ["Synthetic, clean fonts; rules by the authors", "Printed English only; no handwriting or tables",
                                              "Low-DPI binarisation needs upscale first", "Next: real scanned corpora, layout models, auto recipe selection"])
    slide("13. Conclusion", ["Evidence-driven DIP cuts OCR error by about 80%", "Explainable pipeline plus human review", "Thank you. Questions?"])
    prs.save(str(path))


def CH_IMG(n):
    return C.CH / n


# ------------------------------------------------------------------------------ files
def copy_files():
    src = ROOT / "backend"
    picks = ["app/main.py", "app/dip/pipeline.py", "app/dip/metrics.py", "app/dip/steps/basic.py", "app/dip/steps/threshold.py", "app/dip/steps/deskew.py",
             "app/dip/steps/geometry.py", "app/dip/steps/morphology.py", "app/ocr/tesseract_engine.py", "app/classification/keyword_classifier.py",
             "app/classification/ml_classifier.py", "app/classification/classifier.py", "app/extraction/extractors.py", "app/extraction/normalizers.py",
             "app/validation/rules.py", "app/validation/engine.py", "app/services/pipeline_service.py", "app/routers/documents.py",
             "scripts/dataset/build_dataset.py", "scripts/dataset/degrade.py", "scripts/train_classifier.py", "scripts/evaluation/run_experiments.py",
             "scripts/evaluation/make_charts.py", "requirements.txt"]
    for rel in picks:
        p = src / rel
        if p.exists():
            shutil.copy(p, OUT / f"{PREFIX}src_{rel.replace('/', '_')}")
    fe = ROOT / "frontend" / "src"
    for rel in ["app/page.tsx", "app/upload/page.tsx", "components/studio-live.tsx", "components/ocr-live.tsx", "components/fields-live.tsx",
                "components/validation-live.tsx", "components/document-detail.tsx", "lib/api.ts"]:
        p = fe / rel
        if p.exists():
            shutil.copy(p, OUT / f"{PREFIX}src_frontend_{rel.replace('/', '_')}")
    shutil.copy(ROOT / "dataset" / "synthetic" / "manifest.csv", OUT / f"{PREFIX}dataset_manifest.csv")
    for n in ["insurance_policy_00", "claim_form_00", "medical_bill_00"]:
        shutil.copy(ROOT / "dataset" / "synthetic" / "clean" / f"{n}.png", OUT / f"{PREFIX}dataset_sample_{n}.png")
        shutil.copy(ROOT / "dataset" / "synthetic" / "truth" / f"{n}.json", OUT / f"{PREFIX}dataset_truth_{n}.json")
    shutil.copy(ROOT / "evaluation_results" / "RESULTS.md", OUT / f"{PREFIX}results_RESULTS.md")
    shutil.copy(ROOT / "evaluation_results" / "exp1_ablation.csv", OUT / f"{PREFIX}results_exp1_ablation.csv")
    for n in ["dashboard", "upload", "history", "detail", "validation", "evaluation"]:
        shutil.copy(C.IMG / f"{n}.png", OUT / f"{PREFIX}dashboard_{n}.png")
    for n in ["01_ablation_bar", "04_skew", "08_confusion", "09_fields_heatmap", "10_timing"]:
        shutil.copy(C.CH / f"{n}.png", OUT / f"{PREFIX}results_chart_{n}.png")
    shutil.copy(ROOT / "README.md", OUT / f"{PREFIX}README.md")


if __name__ == "__main__":
    build_ieee(OUT / f"{PREFIX}IEEE_4page_Writeup.pdf")
    build_full_report(OUT / f"{PREFIX}Full_Report.pdf")
    build_components()
    build_ppt(OUT / f"{PREFIX}Presentation.pptx")
    copy_files()
    print("done ->", OUT)
