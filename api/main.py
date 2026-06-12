"""FastAPI 应用入口。"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers import chat, documents
from config.settings import settings
from db.database import init_db

# ====== Loguru 配置 ======
logger.remove()  # 移除默认 handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    level="INFO",
)
logger.add(
    str(Path(settings.LOG_DIR) / "app.log"),
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    logger.info("正在启动知识库问答系统...")

    # 创建必要目录
    for d in [settings.UPLOAD_DIR, settings.CHROMA_PERSIST_DIR, settings.LOG_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 初始化数据库
    init_db()
    logger.info("数据库已初始化")

    # 预加载 Embedding 模型
    logger.info("正在加载 Embedding 模型...")
    from core.embeddings import get_embeddings
    get_embeddings()
    logger.info("Embedding 模型已就绪")

    logger.info("系统启动完成 ✓（Reranker 将在首次检索时自动加载）")
    yield
    logger.info("系统已关闭")


app = FastAPI(
    title="智能知识库问答 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
