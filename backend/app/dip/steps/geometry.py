"""Edges, contours, document boundary and perspective transform (techniques 10-13)."""
import cv2
import numpy as np


def _small(img: np.ndarray, height: int = 800) -> tuple[np.ndarray, float]:
    h = img.shape[0]
    if h <= height:
        return img, 1.0
    s = height / h
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


# --- 10. Canny edge detection ------------------------------------------------
def canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with automatic thresholds derived from the median intensity."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    return cv2.Canny(blur, int(max(0, (1 - sigma) * med)), int(min(255, (1 + sigma) * med)))


# --- 11. Contour detection ----------------------------------------------------
def find_contours(edges: np.ndarray):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def draw_contours(base_bgr: np.ndarray, contours, top: int = 15) -> np.ndarray:
    out = base_bgr.copy()
    cv2.drawContours(out, contours[:top], -1, (0, 160, 0), 2)
    return out


# --- 12. Document boundary detection --------------------------------------------
def find_document_quad(bgr: np.ndarray, min_area_frac: float = 0.2, max_area_frac: float = 0.97):
    """Looks for a large 4-corner contour (the paper). Returns corners in the ORIGINAL
    image coordinates (float32, shape 4x2) or None. A quad covering almost the whole
    frame is rejected: the image is already a flat scan, nothing to correct."""
    small, scale = _small(bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)  # close gaps in the outline
    area_img = small.shape[0] * small.shape[1]
    for c in find_contours(edges)[:10]:
        area = cv2.contourArea(c)
        if area < min_area_frac * area_img:
            break  # sorted by area, nothing bigger remains
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area <= max_area_frac * area_img:
            return (approx.reshape(4, 2).astype(np.float32)) / scale
    return None


def draw_quad(bgr: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    out = bgr.copy()
    if quad is not None:
        t = max(2, out.shape[1] // 300)
        cv2.polylines(out, [quad.astype(np.int32)], True, (0, 200, 0), t * 2)
        for p in quad:
            cv2.circle(out, tuple(int(v) for v in p), t * 4, (0, 0, 255), -1)
    return out


# --- 13. Perspective transformation -------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def four_point_transform(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = order_points(quad)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    w, h = max(w, 2), max(h, 2)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC)
