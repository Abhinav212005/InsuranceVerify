"""Grayscale, denoising, contrast enhancement and resizing.

Convention used everywhere: images are NumPy uint8 arrays; colour images are BGR
(OpenCV order), grayscale/binary images are 2-D. Binary images are BLACK text (0)
on WHITE background (255), which is what Tesseract expects.
"""
import cv2
import numpy as np


# --- 2. RGB -> grayscale ---------------------------------------------------
def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # 0.299 R + 0.587 G + 0.114 B


# --- 3. Noise removal ------------------------------------------------------
def gaussian_blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Smooths grain/sensor noise. Kernel size must be odd."""
    return cv2.GaussianBlur(gray, (odd(ksize), odd(ksize)), 0)


def median_blur(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Removes salt-and-pepper specks while keeping stroke edges sharp."""
    return cv2.medianBlur(gray, odd(ksize))


def nlm_denoise(gray: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Non-local means: slower, used as an extra comparison in the experiments."""
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


# --- 4. Contrast enhancement -----------------------------------------------
def hist_equalize(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalization (spreads the whole histogram)."""
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization: equalizes per tile and
    limits amplification so noise is not blown up."""
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile))).apply(gray)


# --- 14. Resizing / normalization -------------------------------------------
def resize_to_width_range(img: np.ndarray, min_w: int = 1800, max_w: int = 3000) -> tuple[np.ndarray, float]:
    """Scale so text size suits Tesseract (about 300 DPI on A4 = 2480 px wide).
    Upscaling uses cubic interpolation, downscaling uses area averaging."""
    h, w = img.shape[:2]
    if w < min_w:
        scale, interp = min(min_w / w, 3.0), cv2.INTER_CUBIC
    elif w > max_w:
        scale, interp = max_w / w, cv2.INTER_AREA
    else:
        return img, 1.0
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    return out, scale


def normalize_minmax(gray: np.ndarray) -> np.ndarray:
    """Stretch intensities to the full 0-255 range."""
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1
