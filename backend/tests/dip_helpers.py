"""Synthetic page generators shared by DIP tests (and later by the evaluation scripts)."""
import cv2
import numpy as np

LINES = [
    "NORTHSTAR GENERAL INSURANCE LTD",
    "Policy Schedule and Certificate",
    "Policy Number: POL-2025-001234",
    "Policy Holder: Aarav Sharma",
    "Period of Insurance: 01/04/2025 to 31/03/2026",
    "Annual Premium: Rs. 12,450.00",
    "Sum Insured: Rs. 5,00,000.00",
    "Vehicle: Honda City MH12AB1234",
    "This policy is subject to the terms and conditions",
    "printed overleaf and the declarations above.",
]


def make_page(w=1200, h=1600, lines=LINES, scale=1.0, seed=0) -> np.ndarray:
    """White page, black text lines (BGR)."""
    img = np.full((h, w, 3), 255, np.uint8)
    for i, text in enumerate(lines * 2):
        y = 120 + i * 70
        if y > h - 80:
            break
        cv2.putText(img, text, (80, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def rotate_img(img: np.ndarray, angle: float, border=255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    b = (border,) * 3 if img.ndim == 3 else border
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=b)


def add_gaussian_noise(img: np.ndarray, sigma=20, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, amount=0.02, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def uneven_light(img: np.ndarray, strength=0.7) -> np.ndarray:
    """Left-to-right illumination gradient (like a shadow across the page)."""
    h, w = img.shape[:2]
    ramp = np.linspace(1 - strength, 1.0, w, dtype=np.float32)[None, :]
    if img.ndim == 3:
        ramp = ramp[..., None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def photo_of_page(page: np.ndarray, out_size=(1400, 1800), margin=170) -> tuple[np.ndarray, np.ndarray]:
    """Paste the page onto a dark desk with a perspective warp. Returns (photo, true corners)."""
    h, w = page.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    W, H = out_size
    dst = np.array([[margin + 40, margin], [W - margin, margin + 60], [W - margin - 50, H - margin], [margin, H - margin - 40]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    desk = np.full((H, W, 3), 45, np.uint8)
    warped = cv2.warpPerspective(page, m, (W, H), borderValue=(45, 45, 45))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (W, H))
    desk[mask > 0] = warped[mask > 0]
    return desk, dst
