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
    'Insurance offices receive policies, claim forms, ID proofs and bills as scans, phone photos and PDFs, and the poor quality of these inputs makes direct OCR unreliable. This paper describes a locally hosted system built around a modular OpenCV preprocessing chain (grayscale conversion, median denoising, adaptive thresholding, projection-profile deskew, morphological clean-up, resizing, and page-boundary detection with perspective correction). Tesseract reads the cleaned page, after which the system classifies the document, extracts and validates its fields, and presents everything in a web dashboard where a reviewer can correct values. It never approves or rejects a claim. Tested on 48 synthetic documents under 15 controlled degradations, the evidence-selected default recipe reduces mean character error rate (CER) from 15.7% on raw images to 3.2%, a relative drop of 79.8%, and mean word error rate from 20.9% to 11.8%. The deskew stage averages 0.04 degrees of angle error, and the page outline was located in all 48 simulated phone photos. End-to-end field accuracy is 88.6%. We also document the steps that made results worse, and note that every figure comes from synthetic data.'
)
KEYWORDS = "digital image processing, optical character recognition, document classification, field extraction, skew correction, adaptive thresholding, OpenCV"

# ----------------------------------------------------------------------------------------------- problem
PROBLEM = [
    ("p", 'Insurance companies process large numbers of mixed documents every day: policy schedules, claim forms, identity proofs, hospital bills and repair invoices. Clerks read each page and retype policy numbers, names, dates and amounts by hand. This is slow and prone to mistakes, and the input quality is uneven, since pages arrive tilted, blurred, noisy, shadowed, at low resolution, or photographed on a desk.'),
    ("p", '<b>Problem statement.</b> Build and evaluate an automated pipeline that (i) raises document image quality using Digital Image Processing (DIP), (ii) reads the text, (iii) determines the document type, (iv) extracts structured fields, (v) flags missing or doubtful values, and (vi) leaves a human in charge by allowing a reviewer to edit every field.'),
    ("p", '<b>Objectives.</b> (1) A modular OpenCV preprocessing pipeline with 14 techniques and stage-by-stage visualisation; (2) Tesseract OCR with word-level confidence; (3) classification into six document categories; (4) null-safe field extraction; (5) validation producing three statuses (Valid, Needs Review, Incomplete); (6) a dashboard containing an Image Processing Studio; (7) a controlled evaluation that measures OCR quality with and without each DIP step.'),
    ("p", '<b>Scope guard.</b> The system only supports data entry. It makes no claim decisions, and it works exclusively on synthetic data.'),
]

# ----------------------------------------------------------------------------------------------- state of the art
SOTA = [
    ("p", "<b>Binarisation.</b> Otsu's method [1] picks one global threshold by maximising the between-class variance and suits evenly lit scans. Local approaches such as Sauvola [10] and Gaussian adaptive thresholding cope with uneven lighting and shadows, but they are more sensitive to noise in flat areas."),
    ("p", '<b>Contrast and noise.</b> CLAHE [3] caps the contrast gain of histogram equalisation within each tile. A median filter removes salt-and-pepper noise while preserving edges, whereas Gaussian smoothing blurs thin character strokes [8].'),
    ("p", '<b>Skew and geometry.</b> Projection-profile skew estimation dates back to Postl [11]. It is still popular because it is easy to explain and tolerant of noise; Hough-transform and minimum-area-rectangle methods are frequent alternatives. For phone photos, the usual route to the page is Canny edges [2], contour search and a four-point perspective warp [9].'),
    ("p", '<b>OCR and document understanding.</b> Tesseract [4] relies on an LSTM recogniser and returns word boxes with confidences. Benchmarks such as FUNSD [5], SROIE [6] and RVL-CDIP [7] have driven layout-aware deep models, which require large labelled sets and GPUs. For narrow domains, rules and linear models over TF-IDF remain strong, inexpensive and auditable baselines [12].'),
    ("p", '<b>Gap.</b> Many document-intake demonstrations show OCR output without measuring what each preprocessing step actually adds, and without reporting where a step does harm. Our contribution is an evidence-driven and fully explainable pipeline, an ablation over controlled degradations, and a human-review workflow, all of which run on a laptop.'),
]

# ----------------------------------------------------------------------------------------------- dataset
DATA = [
    ("p", '<b>Synthetic corpus.</b> No real personal data is involved. A generator renders 48 documents (6 categories with 8 documents each, using four layouts and four fonts) filled with invented names, numbers and dates, and stores the exact text and field values as ground truth. The categories are insurance policy, claim form, identity document, medical bill, repair invoice and other.'),
    ("p", '<b>Degradations.</b> Every page is degraded under 15 controlled conditions with known parameters: low and high Gaussian noise, salt-and-pepper noise, low and high blur, rotation of 3, 8 and 15 degrees, 150 and 100 DPI, a shadow gradient, faded text, heavy JPEG compression, and a simulated phone photo with perspective on a desk background (the clean page is included). This yields 48 x 15 = 720 images, of which 270 are used in the ablation.'),
    ("p", '<b>Other data.</b> As a sanity check for classification and extraction, 24 independent hand-written text documents (different wording, deliberately difficult cases) were written once and never tuned against. For validation, 360 documents with injected defects (missing fields, expired policy, bad dates and amounts, future incident date) were generated together with clean controls. Public corpora (FUNSD, SROIE, RVL-CDIP) are cited as future robustness tests and were not used.'),
    ("p", '<b>Limitation.</b> The synthetic pages use clean TrueType fonts and the rules were written by the author, so the results describe this system on this benchmark rather than real-world accuracy.'),
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
    ("p", '<b>Architecture.</b> A Next.js dashboard communicates with a FastAPI backend. Each upload is hashed (SHA-256), saved, and handled in a background task while the browser polls for status. Every stage image, metric, OCR word and field is kept in SQLite (with file paths on disk), so the Studio can replay each step.'),
    ("fig", IMG / "flowchart.png", "Fig. 1. End-to-end workflow (rows alternate direction): DIP stages (blue), OCR (yellow), intelligence (green).", 0.92),
    ("p", '<b>DIP techniques (14).</b> Acquisition (EXIF correction, 300 DPI rendering); grayscale; Gaussian and median noise removal; histogram equalisation and CLAHE; global, Otsu and adaptive thresholding; deskew; erosion, dilation, opening and closing; Canny edges; contour detection; page-boundary detection; perspective transformation; resizing and normalisation. The recipes are R0 baseline, R1 basic, R2 enhanced grayscale, R3 standard (default), R4 standard with CLAHE and R5 full (boundary and perspective, for phone photos). Throughout, text is black on a white background.'),
    ("p", '<b>Deskew.</b> The ink mask is rotated through candidate angles, and text lines are horizontal when the variance of the row sums peaks. A coarse 1-degree search is refined by a 0.1-degree search, and the result is accepted only if it clearly outperforms zero rotation (Pseudocode 2).'),
    ("code", PSEUDO_PIPELINE, "Pseudocode 1. Processing pipeline (default recipe R3)."),
    ("code", PSEUDO_SKEW, "Pseudocode 2. Projection-profile skew estimation."),
    ("p", '<b>OCR.</b> Tesseract 5 (LSTM, OEM 3, PSM 3) supplies words, boxes and confidences, and words scoring under 60 are flagged. Digital PDFs that already contain a text layer bypass OCR.'),
    ("p", "<b>Classification.</b> A weighted keyword-and-phrase scorer (fuzzy matching absorbs OCR errors, and a threshold yields 'other') is merged with a TF-IDF and logistic-regression model trained on noisy synthetic text. The hybrid averages the two probabilities and lists the evidence keywords."),
    ("p", "<b>Extraction and validation.</b> Label-anchored regular expressions with normalisers (ISO dates, decimal amounts, O/0 correction in numeric contexts) populate each category's field set, leaving null where a value is missing. Validation rules check mandatory fields, date order, expired policy, future incident date, invalid amounts, low field, OCR and classification confidence, and duplicates (Pseudocode 3). A manual edit keeps the original value and triggers re-validation."),
    ("code", PSEUDO_STATUS, "Pseudocode 3. Review-status decision."),
]

# ----------------------------------------------------------------------------------------------- implementation
IMPL = [
    ("p", '<b>Stack.</b> Python, FastAPI, SQLAlchemy and SQLite; OpenCV, NumPy and scikit-image for DIP and metrics; PyMuPDF for PDF rendering; pytesseract for Tesseract 5; scikit-learn for TF-IDF and logistic regression; jiwer for CER and WER; Next.js, TypeScript, Tailwind CSS and Recharts for the dashboard. The backend has roughly 4,300 lines of code and 201 automated tests.'),
    ("p", '<b>Modules.</b> <i>dip/</i> (steps, recipe runner, quality metrics), <i>ocr/</i>, <i>classification/</i>, <i>extraction/</i>, <i>validation/</i>, <i>services/</i> (pipeline orchestration, reports), <i>routers/</i> (REST API) and <i>scripts/</i> (dataset generator, degradations, experiments, charts). Source files are submitted individually.'),
    ("p", '<b>Dashboard.</b> Pages: Dashboard, Upload, Documents, Document Details (Overview, Image Studio, OCR, Intelligence, Validation), Validation Center and Evaluation. Reports can be downloaded as PDF or JSON, and a banner reminds users that the system only assists data entry.'),
]

# ----------------------------------------------------------------------------------------------- novelty
NOVELTY = [
    ("p", '<b>1. Evidence-selected default pipeline.</b> The default recipe comes from an ablation rather than convention: CLAHE, normally recommended, increased CER on text-on-white pages (12.3% versus 11.6%) and so stays optional (R4), while median denoising, adaptive thresholding, deskew and light morphology together reduce CER by 79.8%.'),
    ("p", '<b>2. Explainable deskew on a local ink mask.</b> Running a projection-profile search on an adaptive-threshold mask keeps shadows from misleading it; the mean angle error is 0.04 degrees, and CER at 8 degrees drops from 0.563 to 0.016.'),
    ("p", '<b>3. Honest negative results.</b> Adaptive thresholding alone increased WER (30.5%); Gaussian 5x5 denoising pushed CER to 23.8%; and the default pipeline degrades at 100 DPI (CER 14.1%) where grayscale with upscaling does not. Reporting these shows when each DIP step helps and when it does not.'),
    ("p", '<b>4. Image Processing Studio.</b> Each stage, its parameters and its quality metrics (sharpness, contrast, noise, skew) are stored and displayed in a before/after view, making the effect of DIP visible to the reviewer.'),
    ("p", '<b>5. Fail-safe geometry.</b> When no page boundary is found, the warp is skipped and a note is raised; the pipeline never crashes.'),
    ("p", "<b>6. Human in control.</b> Edits are tracked (original and corrected value), validation runs again, and the interface says 'Valid (data complete)' and never 'Approved'."),
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
    ("p", '<b>Dashboard.</b> The web application runs on live data. Fig. 2 shows the dashboard with processed documents, status counts and recent activity, and the Image Studio displays the six stages (Fig. 3) for a noisy policy page.'),
    ("fig", IMG / "dashboard.png", "Fig. 2. Dashboard: counts by status and recent activity (a non-synthetic test file name is hidden).", 1.0),
    ("fig", IMG / "stages.png", "Fig. 3. Stages on a noisy page: original, grayscale, median denoise, adaptive threshold, deskew, final (after morphology).", 1.0),
    ("p", '<b>Preprocessing ablation (270 images, 15 conditions).</b> Compared with the raw image, the default recipe cuts CER by 79.8% and WER by 43.5%. Adaptive thresholding on its own hurts WER, and deskew followed by morphology recovers it. CLAHE gives no benefit on these pages.'),
    ("table", T_ABL, "TABLE I. Cumulative ablation, mean over 270 images."),
    ("fig", CH / "01_ablation_bar.png", "Fig. 4. Mean CER and WER per cumulative variant.", 1.0),
    ("p", '<b>Denoising and thresholding.</b> Median filtering keeps structure intact (SSIM 0.94-0.97) and outperforms Gaussian blur for OCR. Global thresholding breaks down entirely on faded text (CER 1.0); adaptive thresholding copes with shadow and faded pages but is weaker under heavy noise (0.219) unless a median filter is applied first.'),
    ("table", T_DEN, "TABLE II. Denoising: fidelity to the clean page and OCR error."),
    ("table", T_THR, "TABLE III. Mean CER by thresholding method."),
    ("p", '<b>Geometry.</b> The mean skew-angle error is 0.04 degrees (maximum 0.5). The page outline was found in 48 of 48 simulated phone photos (mean IoU 0.99 against the true page); photo CER is 0.0016 with boundary and perspective correction versus 0.0014 for the raw image, so on these easy photos the warp brings no measurable improvement.'),
    ("table", T_SKEW, "TABLE IV. Deskew accuracy and its effect on CER."),
    ("p", '<b>Classification, extraction and validation.</b> The keyword scorer is robust, whereas the ML model is accurate only in-distribution and falls to 72-79% on held-out text; the hybrid stays at 100% there. Extraction is perfect on clean text but declines with OCR noise. Review status was right for all 360 defect documents, although the low-confidence field rule reaches only 50% precision (30 false positives).'),
    ("table", T_CLS, "TABLE V. Classification accuracy."),
    ("table", T_FLD, "TABLE VI. Field extraction accuracy."),
    ("p", '<b>Speed.</b> On a laptop the pipeline needs 0.49 s per page at 200 DPI and 0.63 s at 300 DPI, with OCR accounting for about 65% of that time.'),
    ("table", T_TIME, "TABLE VII. Mean processing time per page."),
    ("p", '<b>Weakest fields.</b> Vehicle details (53%), claimant and policy-holder names (70-79%) and ID numbers (81%) score lowest, because their layouts vary the most and OCR errors break the label-anchored patterns.'),
]

CONCL = [
    ("p", 'A modular and explainable DIP pipeline lowers OCR error by about 80% on degraded insurance documents, and the classification, extraction and review workflow around it makes the output practical for staff while keeping a person in control. The ablation identifies which steps matter (median denoising, adaptive thresholding, deskew, morphology) and which do not (CLAHE on clean text, Gaussian blur), and it reveals failure modes at low resolution.'),
    ("p", '<b>Limitations.</b> Synthetic data with clean fonts; rules written by the author; printed English only; no handwriting, tables or layout models; a single-user local app. Public scanned corpora and real documents are required before any statement about real-world accuracy. <b>Future work:</b> layout-aware models, table extraction, resolution-aware recipe selection driven by the quality metrics, multi-language OCR, authentication and queue workers.'),
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
