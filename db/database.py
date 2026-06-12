"""SQLite 数据库引擎和会话管理。"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config.settings import settings


class Base(DeclarativeBase):
    pass


def _ensure_dir():
    Path(settings.SQLITE_DB).parent.mkdir(parents=True, exist_ok=True)


_ensure_dir()

engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB}",
    echo=False,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    """建表（启动时调用一次）。"""
    import db.models  # noqa: F401 — 确保模型被注册
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI 依赖：yield 一个 DB session。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
