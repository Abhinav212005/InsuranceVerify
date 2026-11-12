from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.errors import NotFoundError

router = APIRouter(tags=["files"])


@router.get("/files/{path:path}", include_in_schema=False)
def serve_processed_file(path: str):
    """Serves page images from storage/processed. Resolves the path first so '..'
    tricks cannot reach anything outside that folder."""
    base = (settings.storage_dir / "processed").resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise NotFoundError("File")
    return FileResponse(target)
