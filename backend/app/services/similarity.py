"""Perceptual hashing for near-duplicate detection (re-scans of the same document)."""
import cv2
import numpy as np


def dhash(img: np.ndarray, size: int = 8) -> str:
    """Difference hash: shrink to 9x8 grayscale and record whether each pixel is brighter than
    its right neighbour. Survives rescaling, mild noise and compression. Returns 16 hex chars."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
