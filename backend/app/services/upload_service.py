import hashlib
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AppError

ALLOWED_EXT = {".pdf": "pdf", ".png": "png", ".jpg": "jpg", ".jpeg": "jpg"}
MAGIC = [
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
]


def safe_filename(name: str | None) -> str:
    base = Path((name or "").replace("\\", "/")).name.strip()
    return (base or "unnamed")[:255]


def detect_type(filename: str, head: bytes) -> str:
    """Accept only if BOTH the extension and the file's magic bytes agree."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise AppError("UNSUPPORTED_FILE", f"{filename}: only PDF, PNG, JPG and JPEG are supported", 400)
    actual = next((t for sig, t in MAGIC if head.startswith(sig)), None)
    if actual is None or actual != ALLOWED_EXT[ext]:
        raise AppError("UNSUPPORTED_FILE", f"{filename}: content does not match a {ext} file", 400)
    return actual


def read_limited(upload: UploadFile) -> bytes:
    limit = settings.max_upload_mb * 1024 * 1024
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise AppError("FILE_TOO_LARGE", f"{upload.filename}: larger than {settings.max_upload_mb} MB", 413)
    if not data:
        raise AppError("EMPTY_FILE", f"{upload.filename}: file is empty", 400)
    return data


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_original(doc_id: int, file_type: str, data: bytes) -> str:
    """Stores under uploads/<id>/original.<ext>; returns the path relative to storage_dir.
    The stored name never comes from the client, so path traversal is impossible."""
    ext = "jpg" if file_type == "jpg" else file_type
    rel = Path("uploads") / str(doc_id) / f"original.{ext}"
    target = settings.storage_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return rel.as_posix()
