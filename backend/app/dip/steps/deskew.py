"""Skew detection and rotation correction (technique 8).

Method: projection-profile search. Text lines are horizontal when the row sums of the
ink pixels are most "peaky" (lines and gaps alternate sharply), i.e. when the variance
of the row sums is maximal. We rotate a small copy of the page through candidate angles
(coarse search, then fine search) and keep the angle with the highest variance.
It is robust to noise and easy to explain; no assumption about page borders is needed.
"""
import cv2
import numpy as np


def _ink_mask(img: np.ndarray, width: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    # A LOCAL threshold finds the ink even with shadows or a lighting gradient (a global Otsu
    # threshold would mark a whole shadowed region as ink and fool the profile search).
    img = cv2.medianBlur(img, 3)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)  # ink = white


def _score(bw: np.ndarray, angle: float) -> float:
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return float(np.var(rot.sum(axis=1, dtype=np.float64)))


def estimate_correction(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Angle (degrees, OpenCV convention: positive = counter-clockwise) by which the
    page must be rotated to make text horizontal. Returns 0 when there is no clear text."""
    bw = _ink_mask(gray)
    if np.count_nonzero(bw) < 50:
        return 0.0
    coarse = np.arange(-max_angle, max_angle + 1e-6, 1.0)
    best = max(coarse, key=lambda a: _score(bw, a))
    fine = np.arange(best - 1.0, best + 1.0 + 1e-6, 0.1)
    best = max(fine, key=lambda a: _score(bw, a))
    # Only trust the result if it clearly beats "no rotation".
    if _score(bw, best) <= _score(bw, 0.0) * 1.02:
        return 0.0
    return round(float(best), 2)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre keeping the full canvas (nothing is cropped); the
    exposed corners are filled with white."""
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    is_binary = img.ndim == 2 and np.count_nonzero((img != 0) & (img != 255)) == 0
    out = cv2.warpAffine(
        img, m, (nw, nh),
        flags=cv2.INTER_NEAREST if is_binary else cv2.INTER_CUBIC,
        borderValue=255 if img.ndim == 2 else (255, 255, 255),
    )
    return out


def skew_degrees(gray: np.ndarray) -> float:
    """Tilt of the page (positive = text slopes counter-clockwise). The negative of the
    correction angle."""
    return round(-estimate_correction(gray), 2)
