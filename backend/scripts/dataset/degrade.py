"""Controlled image degradations that mimic real scanning problems.

Every function takes a BGR uint8 page and returns (degraded page, metadata). The same seed always
gives the same result, so experiments are reproducible. Metadata records the true parameters
(e.g. the rotation angle or the page corners) so algorithms can be scored against the truth.
"""
import cv2
import numpy as np

WHITE = (255, 255, 255)


def clean(img, rng, idx=0):
    return img, {}


def noise(img, rng, sigma):
    out = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return out, {"sigma": sigma}


def salt_pepper(img, rng, amount=0.03):
    out = img.copy()
    m = rng.random(img.shape[:2])
    out[m < amount / 2] = 0
    out[m > 1 - amount / 2] = 255
    return out, {"amount": amount}


def blur(img, rng, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma), {"sigma": sigma}


def rotate(img, rng, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=WHITE), {"angle": angle}


def low_resolution(img, rng, factor):
    """Fewer pixels (a lower-DPI scan). The image stays small; nothing is scaled back up."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA), {"factor": factor}


def shadow(img, rng, strength=0.6):
    """Diagonal illumination gradient plus a darker corner, like a hand or lamp shadow."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = 1 - strength * (xx / w * 0.7 + yy / h * 0.3)
    vignette = 1 - 0.25 * (((xx - w) ** 2 + (yy - h) ** 2) ** 0.5) / ((w**2 + h**2) ** 0.5)
    out = np.clip(img.astype(np.float32) * (ramp * vignette)[..., None], 0, 255).astype(np.uint8)
    return out, {"strength": strength}


def faded(img, rng, contrast=0.35):
    """Washed-out print: ink is light grey instead of black."""
    out = (img.astype(np.float32) * contrast + 255 * (1 - contrast) * 0.55 + 120 * (1 - contrast) * 0.45).clip(0, 255).astype(np.uint8)
    return noise(out, rng, 4)[0], {"contrast": contrast}


def jpeg(img, rng, quality=20):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), {"quality": quality}


def photo(img, rng, margin=0.09):
    """Phone photo of the page lying on a dark desk: perspective warp, shadow and sensor noise.
    Returns the true page corners (TL, TR, BR, BL) in the output image."""
    h, w = img.shape[:2]
    W, H = int(w * 1.25), int(h * 1.2)
    mx, my = int(W * margin), int(H * margin)
    j = lambda s: int(rng.integers(-s, s + 1))
    dst = np.array(
        [[mx + 30 + j(25), my + j(25)], [W - mx + j(25), my + 40 + j(25)],
         [W - mx - 40 + j(25), H - my + j(25)], [mx + j(25), H - my - 30 + j(25)]], np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    desk = np.full((H, W, 3), 55, np.uint8)
    warped = cv2.warpPerspective(img, m, (W, H), borderValue=(55, 55, 55))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (W, H))
    desk[mask > 0] = warped[mask > 0]
    out, _ = shadow(desk, rng, 0.35)
    out, _ = noise(out, rng, 6)
    return out, {"corners": dst.round(1).tolist()}


CONDITIONS = {
    "clean": (clean, {}),
    "noise_low": (noise, {"sigma": 10}),
    "noise_high": (noise, {"sigma": 25}),
    "salt_pepper": (salt_pepper, {"amount": 0.03}),
    "blur_low": (blur, {"sigma": 1.5}),
    "blur_high": (blur, {"sigma": 3.0}),
    "rot_3": (rotate, {"angle": 3.0}),
    "rot_8": (rotate, {"angle": 8.0}),
    "rot_15": (rotate, {"angle": 15.0}),
    "lowres_150": (low_resolution, {"factor": 0.75}),
    "lowres_100": (low_resolution, {"factor": 0.5}),
    "shadow": (shadow, {"strength": 0.6}),
    "faded": (faded, {"contrast": 0.35}),
    "jpeg_low": (jpeg, {"quality": 20}),
    "photo": (photo, {}),
}


def apply(name: str, img: np.ndarray, doc_index: int, seed: int = 0):
    """Apply a named condition. Rotations alternate direction between documents."""
    fn, params = CONDITIONS[name]
    params = dict(params)
    if fn is rotate and doc_index % 2:
        params["angle"] = -params["angle"]
    rng = np.random.default_rng(seed * 100003 + doc_index * 131 + sum(map(ord, name)))
    return fn(img, rng, **params)
