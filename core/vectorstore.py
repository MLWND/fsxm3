"""Chroma 向量库管理。"""

import uuid
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config.settings import settings
from core.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def add_documents(chunks: list[Document], document_id: str | None = None) -> list[str]:
    """将分块文档添加到向量库，返回 id 列表。"""
    if not chunks:
        return []

    vs = get_vectorstore()
    ids = [str(uuid.uuid4()) for _ in chunks]
    vs.add_documents(documents=chunks, ids=ids)
    logger.info("已添加 {} 个 chunks 到 Chroma", len(chunks))
    return ids


def delete_by_document_id(document_id: str) -> bool:
    """按 document_id 删除所有相关 chunks，返回是否成功。"""
    vs = get_vectorstore()
    try:
        collection = vs._collection

        # 查询匹配的 chunks
        results = collection.get(where={"document_id": document_id})
        matched_ids = results.get("ids", []) if results else []

        if not matched_ids:
            logger.warning("Chroma 中未找到 document_id={} 的 chunks", document_id)
            return True

        # 执行删除
        collection.delete(ids=matched_ids)

        # 验证删除
        verify = collection.get(ids=matched_ids)
        remaining = len(verify.get("ids", []) if verify else [])
        if remaining > 0:
            logger.error("Chroma 删除不完整: document_id={}, 剩余 {} 条", document_id, remaining)
            return False

        logger.info("已从 Chroma 删除 {} 个 chunks (document_id={})", len(matched_ids), document_id)
        return True

    except Exception as e:
        logger.error("Chroma 删除异常: document_id={}, error={}", document_id, e)
        return False


def count_documents() -> int:
    """返回向量库中的总 chunk 数量。"""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0
