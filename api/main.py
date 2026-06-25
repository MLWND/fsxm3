"""FastAPI 应用入口。"""

import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routers import chat, documents
from config.settings import settings
from db.database import init_db


# ====== Loguru 配置 ======
logger.remove()  # 移除默认 handler

# 控制台：人类可读
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    level="INFO",
)

# 文件：结构化 JSON 日志
logger.add(
    str(Path(settings.LOG_DIR) / "app.jsonl"),
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
    format="{message}",
    serialize=True,
)

# 文件：传统文本日志（方便人读）
logger.add(
    str(Path(settings.LOG_DIR) / "app.log"),
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)


def _cleanup_orphan_chunks():
    """清理 Chroma 中无 document_id 或对应 SQLite 文档已不存在的孤儿 chunks。"""
    try:
        from core.vectorstore import get_vectorstore
        from db.database import SessionLocal
        from db.models import Document as DocModel

        vs = get_vectorstore()
        collection = vs._collection
        if collection.count() == 0:
            return

        results = collection.get(include=["metadatas"])
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])

        # 收集 SQLite 中所有有效的 document_id
        db = SessionLocal()
        try:
            valid_ids = {row[0] for row in db.query(DocModel.id).all()}
        finally:
            db.close()

        # 找出孤儿 chunks
        orphan_ids = []
        for chunk_id, meta in zip(ids, metadatas):
            doc_id = (meta or {}).get("document_id")
            if not doc_id or doc_id not in valid_ids:
                orphan_ids.append(chunk_id)

        if orphan_ids:
            collection.delete(ids=orphan_ids)
            logger.info("已清理 {} 个孤儿 chunks（无有效文档关联）", len(orphan_ids))
        else:
            logger.info("未发现孤儿 chunks")

    except Exception as e:
        logger.warning("孤儿 chunks 清理失败: {}", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    logger.info("正在启动知识库问答系统...")

    # 打印关键配置
    device = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"
    logger.info("设备: {} | API Key: {}... | CORS: {}",
                device,
                settings.ZHIPUAI_API_KEY[:8] + "***" if settings.ZHIPUAI_API_KEY else "(空)",
                settings.CORS_ORIGINS)

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

    # 清理 Chroma 中的孤儿数据
    _cleanup_orphan_chunks()

    # 构建 BM25 索引
    from core.hybrid_retriever import rebuild_bm25_index
    rebuild_bm25_index()

    logger.info("系统启动完成 ✓（Reranker 将在首次检索时自动加载）")
    yield
    logger.info("系统已关闭")


app = FastAPI(
    title="智能知识库问答 API",
    version="2.0.0",
    lifespan=lifespan,
)


# ====== 全局异常处理（结构化错误响应）======
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("未捕获异常: request_id={} path={} error={}", request_id, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
        },
    )


# ====== 请求追踪中间件 ======
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.time()

    response = await call_next(request)

    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "request_id={} {} {} → {} ({}ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ====== API Key 认证中间件 ======
@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    # 只在配置了 API_KEY 时启用认证
    if not settings.API_KEY:
        return await call_next(request)

    # 放行健康检查和 docs
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    # 放行 OPTIONS 预检
    if request.method == "OPTIONS":
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == settings.API_KEY:
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={"detail": "无效的 API Key，请在 Authorization 头中提供 Bearer <api_key>"},
    )


# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== 健康检查 ======
@app.get("/health")
def health_check():
    """健康检查端点：用于 Docker/负载均衡器/监控探活。"""
    from core.vectorstore import count_documents
    from db.database import SessionLocal
    from db.models import Document

    db = SessionLocal()
    try:
        doc_count = db.query(Document).count()
    finally:
        db.close()

    chunk_count = count_documents()

    return {
        "status": "healthy",
        "version": "2.0.0",
        "documents": doc_count,
        "chunks": chunk_count,
    }


# ====== 路由注册 ======
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
