from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base


def create_db_engine(url: str) -> Engine:
    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        kwargs["poolclass"] = StaticPool  # one shared in-memory DB (tests)
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):  # SQLite ignores foreign keys unless enabled per connection
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


engine = create_db_engine(settings.db_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# Columns added after the first release. create_all() never alters existing tables, so
# databases created by an older version get these via ALTER TABLE (SQLite supports ADD COLUMN).
ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "documents": {"classification_json": "JSON"},
}


def _ensure_columns(bind: Engine) -> None:
    insp = inspect(bind)
    with bind.begin() as conn:
        for table, cols in ADDED_COLUMNS.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def init_db(bind: Engine | None = None) -> None:
    from app.db import models  # noqa: F401  (register tables)

    bind = bind or engine
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "uploads").mkdir(exist_ok=True)
    (settings.storage_dir / "processed").mkdir(exist_ok=True)
    Base.metadata.create_all(bind)
    _ensure_columns(bind)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
