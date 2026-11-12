from fastapi import APIRouter, Depends

from app.repositories.document_repo import DocumentRepo
from app.routers.documents import get_repo
from app.schemas.document import StatsSummary

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/summary", response_model=StatsSummary)
def summary(repo: DocumentRepo = Depends(get_repo)):
    return repo.stats()
