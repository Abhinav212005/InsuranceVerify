"""Serves the evaluation results produced by scripts/evaluation/ (summary numbers and chart images)."""
import json

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import BASE_DIR
from app.core.errors import AppError, NotFoundError

router = APIRouter(tags=["evaluation"])
RESULTS_DIR = BASE_DIR.parent / "evaluation_results"


@router.get("/evaluation/summary")
def evaluation_summary():
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        raise AppError(
            "NOT_RUN", "No evaluation results yet. Run scripts/evaluation/run_experiments.py and make_charts.py.", 404
        )
    charts = sorted(p.name for p in (RESULTS_DIR / "charts").glob("*.png")) if (RESULTS_DIR / "charts").exists() else []
    return {"summary": json.loads(path.read_text(encoding="utf-8")), "charts": charts}


@router.get("/evaluation/charts/{name}", include_in_schema=False)
def evaluation_chart(name: str):
    base = (RESULTS_DIR / "charts").resolve()
    target = (base / name).resolve()
    if base not in target.parents or target.suffix != ".png" or not target.is_file():
        raise NotFoundError("Chart")
    return FileResponse(target)
