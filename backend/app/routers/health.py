import pytesseract
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        tess = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # Tesseract missing or misconfigured
        return {"ok": False, "tesseract_version": None, "error": str(exc)}
    return {"ok": True, "tesseract_version": tess}
