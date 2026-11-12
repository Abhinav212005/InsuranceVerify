import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from pathlib import Path
OUT = Path("img"); ST = Path("../../backend/storage/processed/4/page_1")


def flow():
    steps = ["Upload\nPDF/PNG/JPG", "Acquire\n(EXIF, 300 DPI)", "Boundary +\nperspective", "Grayscale",
             "Median\ndenoise", "Adaptive\nthreshold", "Deskew", "Morphology\n+ resize",
             "Tesseract\nOCR", "Classify\n(keyword+ML)", "Extract\nfields", "Validate\n3 statuses",
             "SQLite +\ndashboard"]
    cols = ["#e0e7ff"] * 2 + ["#c7d2fe"] * 6 + ["#fde68a"] + ["#bbf7d0"] * 3 + ["#e2e8f0"]
    fig, ax = plt.subplots(figsize=(3.5, 3.1)); ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    pos = []
    for i in range(13):
        r, c = divmod(i, 4)
        x = 0.5 + (c if r % 2 == 0 else 3 - c)
        pos.append((x, 3.4 - r * 0.95))
    for (x, y), s, c in zip(pos, steps, cols):
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.3), 0.84, 0.6, boxstyle="round,pad=0.02", fc=c, ec="#475569", lw=0.7))
        ax.text(x, y, s, ha="center", va="center", fontsize=5.6)
    for a, b in zip(pos, pos[1:]):
        if a[1] == b[1]:
            d = 0.44 if b[0] > a[0] else -0.44
            ax.add_patch(FancyArrowPatch((a[0] + d, a[1]), (b[0] - d, b[1]), arrowstyle="-|>", mutation_scale=6, color="#334155"))
        else:
            ax.add_patch(FancyArrowPatch((a[0], a[1] - 0.33), (b[0], b[1] + 0.33), arrowstyle="-|>", mutation_scale=6, color="#334155"))
    ax.text(2.0, 0.1, "blue: DIP stages (OpenCV)   yellow: OCR   green: intelligence", color="#334155", fontsize=5.2, ha="center")
    fig.tight_layout(pad=0.1); fig.savefig(OUT / "flowchart.png", dpi=300); plt.close(fig)


def montage():
    names = [("original.png", "Original"), ("standard/gray.png", "Grayscale"), ("standard/denoised.png", "Denoised"),
             ("standard/thresholded.png", "Adaptive threshold"), ("standard/deskewed.png", "Deskewed"), ("standard/final.png", "Final")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 2.9)); axs = axs.ravel()
    for ax, (f, t) in zip(axs, names):
        im = Image.open(ST / f).convert("L"); w, h = im.size
        im = im.crop((int(w * 0.02), int(h * 0.055), int(w * 0.58), int(h * 0.265)))
        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=6); ax.axis("off")
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "stages.png", dpi=300); plt.close(fig)


flow(); montage()
