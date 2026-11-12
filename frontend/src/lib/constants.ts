import type { DocCategory, ReviewStatus, StageInfo } from "@/types/api";

export const CATEGORY_LABEL: Record<DocCategory, string> = {
  insurance_policy: "Insurance Policy",
  claim_form: "Claim Form",
  identity_document: "Identity Document",
  medical_bill: "Medical Bill",
  repair_invoice: "Repair Invoice",
  other: "Other Supporting",
};

export const CATEGORY_COLOR: Record<DocCategory, string> = {
  insurance_policy: "#4f46e5",
  claim_form: "#0891b2",
  identity_document: "#7c3aed",
  medical_bill: "#059669",
  repair_invoice: "#d97706",
  other: "#64748b",
};

export const STATUS_LABEL: Record<ReviewStatus, string> = {
  valid: "Valid",
  needs_review: "Needs Review",
  incomplete: "Incomplete",
};

export const STAGES: StageInfo[] = [
  {
    key: "original",
    label: "Original",
    technique: "Image acquisition",
    opencv: "cv2.imdecode / PyMuPDF get_pixmap(dpi=300)",
    explanation:
      "The file is decoded into a pixel array. PDF pages are rendered at 300 DPI because OCR accuracy drops on low-resolution text.",
  },
  {
    key: "gray",
    label: "Grayscale",
    technique: "RGB to grayscale",
    opencv: "cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
    explanation:
      "Three colour channels are merged into one intensity channel (0.299R + 0.587G + 0.114B). Thresholding and OCR only need brightness, and colour stamps stop interfering.",
  },
  {
    key: "denoised",
    label: "Denoised",
    technique: "Gaussian + Median blur, CLAHE",
    opencv: "cv2.GaussianBlur, cv2.medianBlur, createCLAHE().apply",
    explanation:
      "Gaussian blur smooths grain; median blur removes salt-and-pepper specks while keeping edges. CLAHE then boosts local contrast for faded or unevenly lit text.",
  },
  {
    key: "thresholded",
    label: "Thresholded",
    technique: "Otsu / adaptive thresholding",
    opencv: "cv2.threshold(..., THRESH_OTSU), cv2.adaptiveThreshold",
    explanation:
      "The image becomes pure black and white. Otsu picks one best global threshold; adaptive thresholding picks one per neighbourhood, which handles shadows on phone photos.",
  },
  {
    key: "deskewed",
    label: "Deskewed",
    technique: "Skew detection and rotation",
    opencv: "cv2.minAreaRect, cv2.getRotationMatrix2D, cv2.warpAffine",
    explanation:
      "The dominant text angle is estimated and the page is rotated so lines are horizontal. Tesseract degrades quickly beyond a few degrees of tilt.",
  },
  {
    key: "final",
    label: "Final",
    technique: "Morphology + resize/normalize",
    opencv: "cv2.morphologyEx(OPEN/CLOSE), cv2.resize(INTER_CUBIC)",
    explanation:
      "Small specks are removed (opening) and broken strokes are reconnected (closing). The page is scaled so text height suits Tesseract (about 300 DPI).",
  },
];

/** Server-side pipeline stages (GET /documents/{id}/status) in plain language. */
export const PIPELINE_STAGE_LABEL: Record<string, string> = {
  uploaded: "Queued",
  rendering: "Rendering pages",
  rendered: "Pages rendered",
  preprocessing: "Image processing",
  preprocessed: "Image processing done",
  ocr: "Reading text (OCR)",
  ocr_done: "OCR complete",
  ocr_failed: "Done (OCR needs a retry)",
  classifying: "Classifying",
  classified: "Classified",
  extracting: "Extracting fields",
  extracted: "Fields extracted",
  validating: "Validating",
  done: "Done",
};
