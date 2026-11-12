"""Single source of truth for all written deliverables (IEEE paper, full report, component PDFs, PPT).

Block types: ("p", text) ("fig", path, caption, width_fraction) ("table", rows, caption)
             ("code", text, caption) ("h", text)  -- a ("long", block) is only used in the full report.
"""
from pathlib import Path

B = Path(__file__).parent
IMG = B / "img"
CH = B.parent.parent / "evaluation_results" / "charts"

TITLE = "Insurance Document Intelligence: A Digital Image Processing Pipeline for Automated Policy and Claim Intake"

ABSTRACT = (
    "Insurers receive policies, claim forms, identity proofs and bills as scans, phone photos and PDFs whose poor quality "
    "makes plain OCR unreliable. We present a locally run system that applies a modular OpenCV pipeline (grayscale, median "
    "denoising, adaptive thresholding, projection-profile deskew, morphology, resizing, and page-boundary detection with perspective "
    "correction), reads the result with Tesseract, classifies the document, extracts fields, validates them and lets a person review "
    "and correct the output in a web dashboard. The system never approves or rejects a claim. On 48 synthetic documents under "
    "15 controlled degradations, the evidence-selected default recipe lowers mean character error rate (CER) from 15.7% on the raw "
    "image to 3.2% (a 79.8% reduction) and mean word error rate from 20.9% to 11.8%. Deskew has a mean angle error of 0.04 degrees; "
    "page boundaries were found in 48 of 48 phone-style photos. Field accuracy is 88.6% end to end. We also report where steps hurt, "
    "and we stress that all numbers come from synthetic data."
)
KEYWORDS = "digital image processing, OCR, document classification, information extraction, deskew, adaptive thresholding, OpenCV"

# ----------------------------------------------------------------------------------------------- problem
PROBLEM = [
    ("p", "Insurers handle large volumes of mixed documents: policy schedules, claim forms, identity proofs, hospital bills and repair "
          "invoices. Staff read each document and retype policy numbers, names, dates and amounts. The work is slow and error-prone, and "
          "quality varies widely: pages arrive skewed, blurred, noisy, shadowed, low-resolution, or photographed on a desk."),
    ("p", "<b>Problem statement.</b> Design and evaluate an automated pipeline that (i) improves document image quality with Digital "
          "Image Processing (DIP), (ii) extracts text, (iii) identifies the document type, (iv) pulls out structured fields, (v) flags "
          "missing or suspicious values, and (vi) keeps a human in control by letting a reviewer correct every field."),
    ("p", "<b>Objectives.</b> (1) a modular OpenCV preprocessing pipeline covering 14 techniques with stage-by-stage visualisation; "
          "(2) Tesseract OCR with word-level confidence; (3) classification into six categories; (4) null-safe field extraction; "
          "(5) validation with three statuses (Valid, Needs Review, Incomplete); (6) a dashboard with an Image Processing Studio; "
          "(7) a scientific evaluation that measures OCR with and without each DIP step."),
    ("p", "<b>Scope guard.</b> The system assists data entry. It does not approve or reject claims, and it uses only synthetic data."),
]

# ----------------------------------------------------------------------------------------------- state of the art
SOTA = [
    ("p", "<b>Binarisation.</b> Otsu's method [1] chooses a global threshold by maximising between-class variance and works on evenly "
          "lit scans. Local methods such as Sauvola [10] and Gaussian adaptive thresholding handle uneven illumination and shadows, at "
          "the cost of noise sensitivity on flat regions."),
    ("p", "<b>Contrast and noise.</b> CLAHE [3] limits the contrast amplification of histogram equalisation per tile. Median filtering "
          "removes salt-and-pepper noise while keeping edges, whereas Gaussian smoothing blurs thin strokes [8]."),
    ("p", "<b>Skew and geometry.</b> Projection-profile skew estimation goes back to Postl [11] and remains attractive because it is "
          "explainable and robust to noise; Hough-transform and minimum-area-rectangle variants are common alternatives. Page "
          "boundary detection with Canny edges [2], contours and a four-point perspective warp is the standard approach for phone photos [9]."),
    ("p", "<b>OCR and document understanding.</b> Tesseract [4] uses an LSTM recogniser and gives word boxes and confidences. Benchmarks "
          "such as FUNSD [5], SROIE [6] and RVL-CDIP [7] drive layout-aware deep models, which need large labelled data and GPUs. "
          "Rule-based and linear models over TF-IDF remain strong, cheap and auditable baselines for narrow domains [12]."),
    ("p", "<b>Gap.</b> Many intake demos show OCR output but do not measure what each preprocessing step contributes, nor report "
          "where steps hurt. Our contribution is an evidence-driven, fully explainable pipeline with an ablation on controlled "
          "degradations and a human-review workflow, all runnable on a laptop."),
]

# ----------------------------------------------------------------------------------------------- dataset
DATA = [
    ("p", "<b>Synthetic corpus.</b> No real personal data is used. A generator renders 48 documents (6 categories x 8 documents; four layouts "
          "and four fonts) with invented names, numbers and dates, and records the exact text and field values as ground truth. "
          "Categories: insurance policy, claim form, identity document, medical bill, repair invoice and other."),
    ("p", "<b>Degradations.</b> Each page is degraded under 15 controlled conditions with known parameters: low and high Gaussian "
          "noise, salt-and-pepper noise, low and high blur, rotation by 3, 8 and 15 degrees, 150 and 100 DPI, a shadow gradient, "
          "faded text, heavy JPEG compression, and a simulated phone photo with perspective on a desk background (clean included). "
          "This gives 48 x 15 = 720 images; the ablation uses 270 of them."),
    ("p", "<b>Other data.</b> A separate set of 24 independently hand-written text documents (different wording, deliberately hard "
          "cases) was written once and not tuned against, as a sanity check for classification and extraction. For validation, "
          "360 documents with injected defects (missing fields, expired policy, bad dates and amounts, future incident date) and clean "
          "controls were generated. Public corpora (FUNSD, SROIE, RVL-CDIP) are cited as future robustness tests but were not used."),
    ("p", "<b>Limitation.</b> Synthetic pages use clean TrueType fonts, and the rules were written by the authors, so results measure this "
          "system on this benchmark, not real-world accuracy."),
]

# ----------------------------------------------------------------------------------------------- methodology
PSEUDO_PIPELINE = """\
PROCESS(file, recipe = R3):
  pages <- RENDER(file, dpi = 300)            # PyMuPDF / EXIF fix
  for page in pages:
    img <- page
    if recipe has boundary:
        quad <- FIND_QUAD(img)                # blur, Canny, dilate,
        if quad found: img <- WARP(img, quad) #  contours, approxPoly
        else: note "boundary not found"       # never crash
    g <- GRAY(img)
    g <- MEDIAN(g, 3)                         # noise-aware kernel
    b <- ADAPTIVE_THRESH(g, block = 31, C)    # Gaussian, local
    a <- ESTIMATE_SKEW(b);  g <- ROTATE(g, a)
    b <- THRESHOLD(g) ; b <- OPEN/CLOSE(b, 2x2)
    b <- RESIZE(b, text height ~ 20-40 px)
    words <- TESSERACT(b, psm = 3)            # text, boxes, conf
  text <- JOIN(words) ; cls <- CLASSIFY(text)
  F <- EXTRACT(text, cls) ; V <- VALIDATE(F)
  STORE(F, V, stages)"""

PSEUDO_SKEW = """\
ESTIMATE_SKEW(gray, max = 15):
  bw <- ADAPTIVE_THRESH_INV(MEDIAN(small(gray)))
  if INK(bw) < 50: return 0
  best <- argmax over a in [-15, 15] step 1 of
            VAR(ROWSUM(ROTATE(bw, a)))        # peaky profile
  best <- argmax over a in best +/- 1, step 0.1
  if SCORE(best) <= 1.02 * SCORE(0): return 0
  return best"""

PSEUDO_STATUS = """\
STATUS(fields, issues):
  if any mandatory field is missing: return INCOMPLETE
  if any warning (expired, low conf, bad value,
                  duplicate, ...):    return NEEDS_REVIEW
  return VALID                        # "data complete"
                                      # never "approved\""""

METHOD = [
    ("p", "<b>Architecture.</b> A Next.js dashboard talks to a FastAPI backend. An upload is hashed (SHA-256), stored, and processed in a "
          "background task; the browser polls its status. Every stage image, metric, OCR word and field is stored in SQLite "
          "(file paths on disk), so the Studio can replay each step."),
    ("fig", IMG / "flowchart.png", "Fig. 1. End-to-end workflow (rows alternate direction): DIP stages (blue), OCR (yellow), intelligence (green).", 0.92),
    ("p", "<b>DIP techniques (14).</b> Acquisition (EXIF fix, 300 DPI rendering); grayscale; Gaussian and median noise removal; "
          "histogram equalisation and CLAHE; global, Otsu and adaptive thresholding; deskew; erosion, dilation, opening, closing; "
          "Canny edges; contour detection; page-boundary detection; perspective transformation; resizing and normalisation. "
          "The recipes are R0 baseline, R1 basic, R2 enhanced grayscale, R3 standard (default), R4 standard with CLAHE and R5 full "
          "(boundary and perspective, for phone photos). Polarity convention: black text on white background throughout."),
    ("p", "<b>Deskew.</b> The ink mask is rotated through candidate angles; text lines are horizontal when the variance of row sums is "
          "maximal. A coarse search (1 degree) is followed by a fine search (0.1 degree), and the result is only trusted if it clearly beats "
          "zero rotation (Pseudocode 2)."),
    ("code", PSEUDO_PIPELINE, "Pseudocode 1. Processing pipeline (default recipe R3)."),
    ("code", PSEUDO_SKEW, "Pseudocode 2. Projection-profile skew estimation."),
    ("p", "<b>OCR.</b> Tesseract 5 (LSTM, OEM 3, PSM 3) returns words, boxes and confidences; words below 60 are flagged. Digital PDFs "
          "with a text layer skip OCR."),
    ("p", "<b>Classification.</b> A weighted keyword and phrase scorer (with fuzzy matching for OCR errors and a threshold for 'other') is "
          "combined with a TF-IDF + logistic-regression model trained on noisy synthetic text; the hybrid averages both probabilities and "
          "reports the evidence keywords."),
    ("p", "<b>Extraction and validation.</b> Label-anchored regular expressions with normalisers (ISO dates, decimal amounts, O/0 fixes "
          "in numeric contexts) fill each category's field set, with null for anything missing. Validation rules cover mandatory "
          "fields, date order, expired policy, future incident date, invalid amounts, low field, OCR and classification confidence, "
          "and duplicates (Pseudocode 3). Manual edits keep the original value and re-run validation."),
    ("code", PSEUDO_STATUS, "Pseudocode 3. Review-status decision."),
]

# ----------------------------------------------------------------------------------------------- implementation
IMPL = [
    ("p", "<b>Stack.</b> Python, FastAPI, SQLAlchemy and SQLite; OpenCV, NumPy and scikit-image for DIP and metrics; PyMuPDF for PDF "
          "rendering; pytesseract for Tesseract 5; scikit-learn for TF-IDF and logistic regression; jiwer for CER and WER; Next.js, "
          "TypeScript, Tailwind CSS and Recharts for the dashboard. About 4,300 lines of backend code and 201 automated tests."),
    ("p", "<b>Modules.</b> <i>dip/</i> (steps, recipe runner, quality metrics), <i>ocr/</i>, <i>classification/</i>, <i>extraction/</i>, "
          "<i>validation/</i>, <i>services/</i> (pipeline orchestration, reports), <i>routers/</i> (REST API) and <i>scripts/</i> "
          "(dataset generator, degradations, experiments, charts). Source files are submitted individually."),
    ("p", "<b>Dashboard.</b> Pages: Dashboard, Upload, Documents, Document Details (Overview, Image Studio, OCR, Intelligence, Validation), "
          "Validation Center and Evaluation. Reports download as PDF or JSON. A banner states that the system assists data entry only."),
]

# ----------------------------------------------------------------------------------------------- novelty
NOVELTY = [
    ("p", "<b>1. Evidence-selected default pipeline.</b> The default recipe was chosen from an ablation, not from convention: CLAHE, usually "
          "recommended, raised CER on text-on-white pages (12.3% versus 11.6%) and is therefore optional (R4), while median denoising, "
          "adaptive thresholding, deskew and light morphology together cut CER by 79.8%."),
    ("p", "<b>2. Explainable deskew with a local ink mask.</b> A projection-profile search on an adaptive-threshold mask is not fooled by "
          "shadows; mean angle error is 0.04 degrees, and CER at 8 degrees falls from 0.563 to 0.016."),
    ("p", "<b>3. Honest negative results.</b> Adaptive thresholding alone raised WER (30.5%); Gaussian 5x5 denoising raised CER to 23.8%; the "
          "default pipeline degrades at 100 DPI (CER 14.1%) where grayscale with upscaling does not. Reporting these shows when each DIP "
          "step helps."),
    ("p", "<b>4. Image Processing Studio.</b> Every stage, its parameters and quality metrics (sharpness, contrast, noise, skew) are stored and "
          "shown with before/after comparison, so the DIP effect is visible to the reviewer."),
    ("p", "<b>5. Fail-safe geometry.</b> If no page boundary is found the warp is skipped and flagged; the pipeline never crashes."),
    ("p", "<b>6. Human in control.</b> Edits are tracked (original and corrected value), validation re-runs, and the interface says "
          "'Valid (data complete)', never 'Approved'."),
]

# ----------------------------------------------------------------------------------------------- results
T_ABL = [["Variant (cumulative)", "CER %", "WER %"],
         ["A0 raw image", "15.7", "20.9"], ["A1 + grayscale", "12.0", "17.3"], ["A2 + median denoise", "11.6", "16.0"],
         ["A3 + CLAHE", "12.3", "16.6"], ["A4 + adaptive threshold", "16.2", "30.5"], ["A5 + deskew", "6.5", "20.7"],
         ["A6 + morphology", "3.9", "15.1"], ["R3 default (no CLAHE)", "3.2", "11.8"]]
T_DEN = [["Denoising", "PSNR dB", "SSIM", "CER"],
         ["none", "28.69", "0.645", "0.031"], ["Gaussian 5x5", "29.53", "0.819", "0.238"], ["median 3", "35.57", "0.942", "0.075"],
         ["median 5", "30.73", "0.967", "0.017"], ["non-local means", "29.40", "0.759", "0.030"]]
T_THR = [["Method", "clean", "noise_high", "shadow", "faded"],
         ["global (T=127)", "0.007", "0.008", "0.003", "1.000"], ["Otsu", "0.006", "0.006", "0.012", "0.006"],
         ["adaptive", "0.003", "0.219", "0.003", "0.003"]]
T_SKEW = [["Condition", "angle error", "CER deskew", "CER none"],
          ["rot 3", "0.05", "0.006", "0.006"], ["rot 8", "0.04", "0.016", "0.563"], ["rot 15", "0.09", "0.134", "0.994"]]
T_CLS = [["Method", "end-to-end (720 img)", "held-out clean", "held-out noisy"],
         ["keyword", "99.2%", "100%", "98.8%"], ["TF-IDF + LogReg", "100%", "78.7%", "72.1%"], ["hybrid", "99.4%", "100%", "100%"]]
T_FLD = [["Test set", "fields", "accuracy"],
         ["synthetic text, clean", "3480", "100.0%"], ["synthetic text, OCR-like noise", "3480", "82.2%"],
         ["images, true category (end to end)", "-", "88.6%"], ["independent hand-written docs", "116", "96.6%"]]
T_TIME = [["Stage (ms/page)", "200 DPI", "300 DPI"],
          ["threshold", "63", "135"], ["deskew", "78", "76"], ["OCR (Tesseract)", "325", "395"], ["total", "492", "634"]]

RESULTS = [
    ("p", "<b>Dashboard.</b> The web application runs on live data. Fig. 2 shows the dashboard with processed documents, the status "
          "counts and recent activity; the Image Studio shows the six stages (Fig. 3) for a noisy policy page."),
    ("fig", IMG / "dashboard.png", "Fig. 2. Dashboard: counts by status and recent activity (a non-synthetic test file name is hidden).", 1.0),
    ("fig", IMG / "stages.png", "Fig. 3. Stages on a noisy page: original, grayscale, median denoise, adaptive threshold, deskew, final (after morphology).", 1.0),
    ("p", "<b>Preprocessing ablation (270 images, 15 conditions).</b> The default recipe lowers CER by 79.8% and WER by 43.5% relative to the raw "
          "image. Adaptive thresholding alone hurts WER; deskew, then morphology, recover it. CLAHE does not help on these pages."),
    ("table", T_ABL, "TABLE I. Cumulative ablation, mean over 270 images."),
    ("fig", CH / "01_ablation_bar.png", "Fig. 4. Mean CER and WER per cumulative variant.", 1.0),
    ("p", "<b>Denoising and thresholding.</b> Median filtering preserves structure (SSIM 0.94-0.97) and beats Gaussian blur for OCR. "
          "Global thresholding fails completely on faded text (CER 1.0); adaptive handles shadow and faded pages but is weaker "
          "under heavy noise (0.219) unless a median filter comes first."),
    ("table", T_DEN, "TABLE II. Denoising: fidelity to the clean page and OCR error."),
    ("table", T_THR, "TABLE III. Mean CER by thresholding method."),
    ("p", "<b>Geometry.</b> Mean skew-angle error is 0.04 degrees (max 0.5). The page outline was found in 48 of 48 simulated phone photos "
          "(mean IoU 0.99 with the true page); photo CER is 0.0016 with boundary and perspective versus 0.0014 for the raw image, "
          "so on these easy photos the warp gives no measurable gain."),
    ("table", T_SKEW, "TABLE IV. Deskew accuracy and its effect on CER."),
    ("p", "<b>Classification, extraction and validation.</b> The keyword scorer is robust; the ML model is accurate only in-distribution "
          "and drops to 72-79% on held-out text, while the hybrid stays at 100% on held-out text. Extraction is perfect on clean text but "
          "falls with OCR noise. Review status was correct for all 360 defect documents; the low-confidence field rule has only "
          "50% precision (30 false positives)."),
    ("table", T_CLS, "TABLE V. Classification accuracy."),
    ("table", T_FLD, "TABLE VI. Field extraction accuracy."),
    ("p", "<b>Speed.</b> The pipeline takes 0.49 s per page at 200 DPI and 0.63 s at 300 DPI on a laptop, of which OCR is about 65%."),
    ("table", T_TIME, "TABLE VII. Mean processing time per page."),
    ("p", "<b>Weakest fields.</b> Vehicle details (53%), claimant and policy-holder names (70-79%) and ID numbers (81%) are weakest, because "
          "their layout varies most and OCR errors break the label-anchored patterns."),
]

CONCL = [
    ("p", "A modular, explainable DIP pipeline reduces OCR error by about 80% on degraded insurance documents, and the surrounding "
          "intelligence and review workflow makes the output usable by staff while keeping a human in control. The ablation shows which "
          "steps matter (median denoising, adaptive thresholding, deskew, morphology) and which do not (CLAHE on clean text; "
          "Gaussian blur), and it exposes failure modes at low resolution."),
    ("p", "<b>Limitations.</b> Synthetic data with clean fonts; rules written by the authors; printed English only; no handwriting, tables or "
          "layout models; single-user local app. Public scanned corpora and real documents are needed before claims about "
          "real-world accuracy. <b>Future work:</b> layout-aware models, table extraction, resolution-aware recipe selection "
          "from the quality metrics, multi-language OCR, authentication and queue workers."),
]

REFS = [
    "N. Otsu, \"A threshold selection method from gray-level histograms,\" IEEE Trans. Syst., Man, Cybern., vol. 9, no. 1, pp. 62-66, 1979.",
    "J. Canny, \"A computational approach to edge detection,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 8, no. 6, pp. 679-698, 1986.",
    "K. Zuiderveld, \"Contrast limited adaptive histogram equalization,\" in Graphics Gems IV, Academic Press, 1994, pp. 474-485.",
    "R. Smith, \"An overview of the Tesseract OCR engine,\" in Proc. ICDAR, 2007, pp. 629-633.",
    "G. Jaume, H. K. Ekenel and J.-P. Thiran, \"FUNSD: A dataset for form understanding in noisy scanned documents,\" in Proc. ICDAR Workshops, 2019.",
    "Z. Huang et al., \"ICDAR2019 competition on scanned receipt OCR and information extraction,\" in Proc. ICDAR, 2019.",
    "A. W. Harley, A. Ufkes and K. G. Derpanis, \"Evaluation of deep convolutional nets for document image classification and retrieval,\" in Proc. ICDAR, 2015.",
    "R. C. Gonzalez and R. E. Woods, Digital Image Processing, 4th ed. Pearson, 2018.",
    "G. Bradski, \"The OpenCV library,\" Dr. Dobb's Journal of Software Tools, 2000.",
    "J. Sauvola and M. Pietikainen, \"Adaptive document image binarization,\" Pattern Recognition, vol. 33, no. 2, pp. 225-236, 2000.",
    "W. Postl, \"Detection of linear oblique structures and skew in digitized documents,\" in Proc. ICPR, 1986, pp. 687-689.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.",
    "V. I. Levenshtein, \"Binary codes capable of correcting deletions, insertions and reversals,\" Soviet Physics Doklady, vol. 10, no. 8, pp. 707-710, 1966.",
]

# Extra, report-only material
LONG_EXTRA = {
    "intro": [
        ("p", "The report follows the required components: problem statement, state of the art, methodology (flowchart and pseudocode), "
              "implementation and source code, dataset, novelty using DIP, results and dashboards, and references. All figures and "
              "numbers are produced by the scripts in the repository (<i>scripts/evaluation</i>) and can be regenerated."),
    ],
    "dip": [
        ("h", "DIP technique reference"),
        ("table", [["#", "Technique", "OpenCV / method", "Why it matters for insurance documents"],
                   ["1", "Acquisition", "imdecode, EXIF transpose, PyMuPDF 300 DPI", "Phone photos carry EXIF rotation; PDFs need 300 DPI"],
                   ["2", "Grayscale", "cvtColor BGR2GRAY", "Neutralises coloured stamps and logos"],
                   ["3", "Noise removal", "GaussianBlur, medianBlur, fastNlMeans", "Fax and photocopy noise; median keeps strokes"],
                   ["4", "Contrast", "equalizeHist, CLAHE", "Faded prints, uneven light (optional recipe R4)"],
                   ["5", "Global threshold", "threshold", "Baseline that fails on shadow and faded text"],
                   ["6", "Adaptive threshold", "adaptiveThreshold (Gaussian)", "Phone photos, shadows; default"],
                   ["7", "Otsu", "THRESH_OTSU", "Clean, evenly lit scans"],
                   ["8", "Deskew", "projection profile, warpAffine", "Tesseract degrades beyond 2-3 degrees"],
                   ["9", "Morphology", "erode, dilate, open, close", "Specks and broken strokes"],
                   ["10", "Canny", "Canny (median-based thresholds)", "Page edges"],
                   ["11", "Contours", "findContours, approxPolyDP", "Page outline candidates"],
                   ["12", "Boundary detection", "largest 4-point contour", "Phone photos; fallback if none"],
                   ["13", "Perspective", "getPerspectiveTransform, warpPerspective", "Flatten the page"],
                   ["14", "Resize / normalise", "resize (cubic / area), normalize", "Text height 20-40 px for OCR"]],
         "TABLE A. The 14 DIP techniques implemented."),
    ],
    "eval": [
        ("h", "Additional results"),
        ("fig", CH / "02_threshold.png", "Fig. A1. Thresholding comparison by condition.", 0.8),
        ("fig", CH / "03_denoise.png", "Fig. A2. Denoising comparison.", 0.8),
        ("fig", CH / "04_skew.png", "Fig. A3. Skew-angle error and effect on CER.", 0.8),
        ("fig", CH / "05_resolution.png", "Fig. A4. Resolution study: the default pipeline degrades at 100 DPI and below; grayscale with upscaling does not.", 0.8),
        ("fig", CH / "08_confusion.png", "Fig. A5. Classification confusion matrix.", 0.7),
        ("fig", CH / "09_fields_heatmap.png", "Fig. A6. Field accuracy by category and field.", 0.9),
        ("fig", CH / "10_timing.png", "Fig. A7. Processing time per stage.", 0.8),
        ("fig", CH / "11_validation.png", "Fig. A8. Validation rule precision and recall.", 0.8),
        ("fig", IMG / "upload.png", "Fig. A9. Upload page.", 0.9),
        ("fig", IMG / "history.png", "Fig. A10. Documents (history) page with filters.", 0.9),
        ("fig", IMG / "detail.png", "Fig. A11. Document details: classification, evidence keywords and OCR confidence.", 0.9),
        ("fig", IMG / "validation.png", "Fig. A12. Validation Center queue.", 0.9),
        ("fig", IMG / "evaluation.png", "Fig. A13. Evaluation page inside the dashboard.", 0.9),
        ("p", "<b>Failure analysis.</b> (i) At 100 DPI and below the default pipeline binarises thin strokes badly; the planned fix is to upscale "
              "before thresholding (grayscale plus upscale gives CER 0.025 at 50 DPI versus 0.782 for the default pipeline). (ii) The low-confidence-field rule "
              "fires on correct values, so its threshold should be tuned on a development split. (iii) ML classification does not "
              "generalise beyond its training vocabulary; the keyword scorer carries the hybrid. (iv) Page-boundary warping showed no "
              "gain on easy photos, and needs harder backgrounds to be evaluated."),
        ("p", "<b>Reproducibility.</b> Run <i>build_dataset</i>, <i>train_classifier</i>, <i>run_experiments</i> and <i>make_charts</i> (see the README). "
              "Seeds are fixed; the test set was not used to select the default recipe except through the ablation, which is a "
              "limitation: a separate development split would be cleaner."),
    ],
}
