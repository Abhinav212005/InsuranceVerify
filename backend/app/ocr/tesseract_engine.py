"""Tesseract OCR wrapper: text, word boxes and per-word confidence.

Tesseract settings used:
  --oem 3   default engine (LSTM neural network)
  --psm N   page segmentation mode (how Tesseract expects the page to be laid out):
             3 fully automatic (default, mixed pages)    4 one column of variable-size text
             6 one uniform block of text (forms, invoices)  11 sparse text, find as much as possible
             12 sparse text + orientation detection
  preserve_interword_spaces=1  keeps spacing between words, which helps column-style forms
"""
import numpy as np
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError

from app.core.config import settings
from app.core.errors import AppError

ALLOWED_PSM = {3: "auto", 4: "single column", 6: "uniform block", 11: "sparse text", 12: "sparse text + OSD"}


def _configure() -> None:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def check_psm(psm: int) -> int:
    if psm not in ALLOWED_PSM:
        raise AppError("INVALID_PSM", f"psm must be one of {sorted(ALLOWED_PSM)}", 422)
    return psm


def recognize(img: np.ndarray, psm: int = 3, lang: str = "eng") -> dict:
    """Runs Tesseract on a grayscale/binary/BGR page image.

    Returns {"words": [...], "text": str, "avg_confidence": float}. Word boxes are fractions
    (0..1) of the image width/height so overlays work at any display size.
    """
    _configure()
    check_psm(psm)
    h, w = img.shape[:2]
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=Output.DICT, timeout=settings.ocr_timeout_s
        )
    except TesseractNotFoundError:
        raise AppError("OCR_UNAVAILABLE", "Tesseract is not installed or TESSERACT_CMD is wrong", 503)
    except RuntimeError:  # pytesseract raises RuntimeError on timeout
        raise AppError("OCR_TIMEOUT", f"OCR took longer than {settings.ocr_timeout_s} s", 504)
    except TesseractError as e:
        raise AppError("OCR_FAILED", f"Tesseract failed: {str(e)[:200]}", 500)

    words = []
    line_index: dict[tuple[int, int, int], int] = {}
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if not text or conf < 0:  # conf -1 marks layout elements that are not words
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line = line_index.setdefault(key, len(line_index))
        words.append(
            {
                "text": text,
                "conf": round(conf, 1),
                "x": round(data["left"][i] / w, 4),
                "y": round(data["top"][i] / h, 4),
                "w": round(data["width"][i] / w, 4),
                "h": round(data["height"][i] / h, 4),
                "line": line,
                "block": key[0],
            }
        )
    return {"words": words, "text": assemble_text(words), "avg_confidence": average_confidence(words)}


def assemble_text(words: list[dict]) -> str:
    """Rebuild page text: words joined by spaces, one line per OCR line, blank line between blocks."""
    out: list[str] = []
    cur_line, cur_block, buf = None, None, []
    for w in words:
        if w["line"] != cur_line:
            if buf:
                out.append(" ".join(buf))
            if cur_block is not None and w["block"] != cur_block:
                out.append("")
            cur_line, cur_block, buf = w["line"], w["block"], []
        buf.append(w["text"])
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)


def average_confidence(words: list[dict]) -> float:
    return round(float(np.mean([w["conf"] for w in words])), 1) if words else 0.0


def low_confidence(words: list[dict], threshold: float) -> list[dict]:
    return [w for w in words if w["conf"] < threshold]
