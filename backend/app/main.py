import logging
from contextlib import asynccontextmanager

import pytesseract
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.errors import register_error_handlers
from app.db.session import init_db
from app.routers import documents, evaluation, files, health, processing, stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Insurance Document Intelligence API", version="0.3.0", lifespan=lifespan)
register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
for r in (health.router, documents.router, processing.router, stats.router, evaluation.router):
    app.include_router(r, prefix=API_PREFIX)

app.include_router(files.router)  # page images at /files/<doc>/page_<n>/...
