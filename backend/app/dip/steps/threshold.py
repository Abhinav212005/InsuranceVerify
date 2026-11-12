"""Binarization: global, Otsu and adaptive thresholding (techniques 5, 6, 7)."""
import cv2
import numpy as np

from app.dip.steps.basic import odd


def global_threshold(gray: np.ndarray, value: int = 127) -> np.ndarray:
    """One fixed cut-off. Works only on evenly lit pages, so it is the baseline."""
    _, out = cv2.threshold(gray, int(value), 255, cv2.THRESH_BINARY)
    return out


def otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Picks the cut-off that maximizes between-class variance. Returns (binary, T)."""
    t, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out, float(t)


def auto_c(gray: np.ndarray) -> float:
    """Offset C proportional to the measured image noise. A fixed C turns background noise
    into black specks on noisy (or contrast-boosted) pages; about 6 sigma (Immerkaer estimate, which under-reads correlated noise) keeps them white."""
    from app.dip.metrics import noise_sigma

    return float(np.clip(6.0 * noise_sigma(gray, normalize=False), 10.0, 40.0))


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: float = 10.0) -> np.ndarray:
    """Threshold per neighbourhood (Gaussian weighted mean - C). Handles shadows and
    uneven lighting. block_size must be odd and about the size of a few characters.
    Pass c <= 0 to choose C automatically from the image noise."""
    if c <= 0:
        c = auto_c(gray)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, odd(max(block_size, 3)), float(c)
    )
