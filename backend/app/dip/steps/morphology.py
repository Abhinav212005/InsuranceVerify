"""Morphological operations (technique 9).

Morphology is defined for WHITE foreground. Our binary pages are black text on white,
so every function inverts first, works on white text, and inverts back. In "text"
terms: erosion thins strokes, dilation thickens them, opening removes specks,
closing reconnects broken strokes.
"""
import cv2
import numpy as np

OPS = ("erode", "dilate", "open", "close")


def kernel(size: int = 2, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    return cv2.getStructuringElement(shape, (max(1, int(size)), max(1, int(size))))


def apply(binary: np.ndarray, op: str, ksize: int = 2) -> np.ndarray:
    if op == "none":
        return binary
    if op not in OPS:
        raise ValueError(f"unknown morphology op: {op}")
    k = kernel(ksize)
    fg = cv2.bitwise_not(binary)  # text -> white
    if op == "erode":
        out = cv2.erode(fg, k)
    elif op == "dilate":
        out = cv2.dilate(fg, k)
    elif op == "open":
        out = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    else:
        out = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    return cv2.bitwise_not(out)
