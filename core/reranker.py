"""Reranker：使用 CrossEncoder 对检索结果精排。"""

from langchain_core.documents import Document
from loguru import logger

from config.settings import settings

_reranker = None
_reranker_loaded = False


def get_reranker():
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        logger.info("正在加载 Reranker 模型: {}", settings.RERANKER_MODEL)
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
        _reranker_loaded = True
        logger.info("Reranker 模型已就绪")
        return _reranker
    except Exception as e:
        logger.warning("Reranker 加载失败，将使用纯检索: {}", e)
        _reranker_loaded = True
        return None


def rerank(query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
    """对检索结果重排序，返回最相关的 top_k 条。"""
    if not documents:
        return []

    top_k = top_k or settings.RERANK_TOP_K
    model = get_reranker()

    if model is None:
        return documents[:top_k]

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(documents, scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, _score in scored_docs[:top_k]]
