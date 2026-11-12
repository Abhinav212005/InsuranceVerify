"""Turns an uploaded file into per-page PNG images (the input of the DIP pipeline)."""
import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.errors import AppError

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


@dataclass
class RenderedPage:
    page_number: int
    original_path: Path
    thumb_path: Path
    width: int
    height: int
    text: str | None  # embedded text layer, if the PDF page has one


def inspect_pdf(data: bytes) -> int:
    """Cheap upload-time check. Returns the page count or raises a clean error."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise AppError("CORRUPT_FILE", "The PDF could not be opened", 400)
    with doc:
        if doc.needs_pass:
            raise AppError("ENCRYPTED_PDF", "Password-protected PDFs are not supported", 400)
        n = doc.page_count
    if n < 1:
        raise AppError("CORRUPT_FILE", "The PDF has no pages", 400)
    if n > settings.max_pages:
        raise AppError("PAGE_LIMIT", f"PDF has {n} pages; the limit is {settings.max_pages}", 400)
    return n


def inspect_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Image.DecompressionBombError:
        raise AppError("IMAGE_TOO_LARGE", "Image resolution is too large", 400)
    except Exception:
        raise AppError("CORRUPT_FILE", "The image could not be decoded", 400)


def _save_thumb(src: Path, dest: Path) -> None:
    with Image.open(src) as im:
        im.thumbnail((400, 560))
        im.convert("RGB").save(dest, "JPEG", quality=80)


def _page_dir(doc_id: int, page_number: int) -> Path:
    d = settings.storage_dir / "processed" / str(doc_id) / f"page_{page_number}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_pdf(doc_id: int, pdf_path: Path) -> list[RenderedPage]:
    pages: list[RenderedPage] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            # Lower the DPI for oversized pages so the longest side stays bounded.
            longest_in = max(page.rect.width, page.rect.height) / 72
            dpi = min(settings.render_dpi, int(settings.max_render_side_px / longest_in))
            pix = page.get_pixmap(dpi=max(dpi, 72), alpha=False)
            out = _page_dir(doc_id, i)
            original, thumb = out / "original.png", out / "thumb.jpg"
            pix.save(original)
            _save_thumb(original, thumb)
            pages.append(
                RenderedPage(
                    i, original, thumb, pix.width, pix.height,
                    text if len(text) >= settings.text_layer_min_chars else None,
                )
            )
    return pages


def render_image(doc_id: int, image_path: Path) -> list[RenderedPage]:
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im)  # phone photos carry rotation in EXIF
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        out = _page_dir(doc_id, 1)
        original, thumb = out / "original.png", out / "thumb.jpg"
        im.save(original, "PNG", compress_level=3)
        w, h = im.size
    _save_thumb(original, thumb)
    return [RenderedPage(1, original, thumb, w, h, None)]
