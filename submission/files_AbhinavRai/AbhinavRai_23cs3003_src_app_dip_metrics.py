"""Image quality measures used to judge whether preprocessing helped.

All measures are computed on a copy resized to a fixed width, so numbers are comparable
between images of different resolution.
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from app.dip.steps.basic import to_gray
from app.dip.steps.deskew import skew_degrees

REF_WIDTH = 1000


def _norm(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if w == REF_WIDTH:
        return gray
    interp = cv2.INTER_AREA if w > REF_WIDTH else cv2.INTER_CUBIC
    return cv2.resize(gray, (REF_WIDTH, max(1, int(h * REF_WIDTH / w))), interpolation=interp)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher = sharper edges."""
    return float(cv2.Laplacian(_norm(gray), cv2.CV_64F).var())


def contrast(gray: np.ndarray) -> float:
    """Standard deviation of intensities."""
    return float(np.std(_norm(gray)))


def noise_sigma(gray: np.ndarray, normalize: bool = True) -> float:
    """Immerkaer's fast noise estimator (standard deviation of noise)."""
    g = (_norm(gray) if normalize else gray).astype(np.float64)
    h, w = g.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def histogram(gray: np.ndarray, bins: int = 32) -> list[int]:
    return np.histogram(gray, bins=bins, range=(0, 256))[0].astype(int).tolist()


def quality(img: np.ndarray, with_skew: bool = False) -> dict:
    gray = to_gray(img)
    out = {
        "sharpness": round(sharpness(gray), 2),
        "contrast": round(contrast(gray), 2),
        "noise": round(noise_sigma(gray), 2),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }
    if with_skew:
        out["skew_deg"] = skew_degrees(gray)
    return out


def psnr_ssim(clean: np.ndarray, test: np.ndarray) -> dict:
    """Needs a clean reference (synthetic documents). Images must have equal size."""
    a, b = to_gray(clean), to_gray(test)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return {
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=255)),
        "ssim": float(structural_similarity(a, b, data_range=255)),
    }
