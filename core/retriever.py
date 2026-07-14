"""检索 pipeline：本地 MMR/Hybrid + 可选 Exa 联网搜索。"""

from typing import Literal

from langchain_core.documents import Document
from loguru import logger

from config.settings import settings
from core.vectorstore import get_vectorstore


def retrieve_local(query: str, top_k: int | None = None, use_hybrid: bool = False) -> list[Document]:
    """本地知识库检索（MMR 或 Hybrid + Rerank）。"""
    top_k = top_k or settings.RERANK_TOP_K

    if use_hybrid:
        from core.hybrid_retriever import get_hybrid_retriever
        retriever = get_hybrid_retriever()
        candidates = retriever.search_hybrid(query, top_k=settings.RETRIEVAL_TOP_K)
    else:
        vs = get_vectorstore()
        candidates = vs.max_marginal_relevance_search(
            query,
            k=settings.RETRIEVAL_TOP_K,
            fetch_k=settings.RETRIEVAL_TOP_K,
            lambda_mult=0.7,
        )

    if not candidates:
        return []

    from core.reranker import rerank
    return rerank(query, candidates, top_k=top_k)


def retrieve(
    query: str,
    top_k: int | None = None,
    use_hybrid: bool = False,
    search_mode: Literal["local", "web", "hybrid"] = "local",
) -> tuple[list[Document], list[Document]]:
    """检索入口。

    Args:
        query: 搜索查询
        top_k: 每路返回结果数
        use_hybrid: 本地是否使用混合检索（BM25 + Vector）
        search_mode: "local" 仅本地 / "web" 仅网络 / "hybrid" 混合

    Returns:
        (local_docs, web_docs)
    """
    top_k = top_k or settings.RERANK_TOP_K

    local_docs: list[Document] = []
    web_docs: list[Document] = []

    if search_mode in ("local", "hybrid"):
        local_docs = retrieve_local(query, top_k=top_k, use_hybrid=use_hybrid)

    if search_mode in ("web", "hybrid"):
        try:
            from core.web_search import web_search
            web_docs = web_search(query, top_k=settings.WEB_SEARCH_TOP_K)
        except ValueError:
            logger.warning("EXA_API_KEY 未配置，跳过联网搜索")
        except RuntimeError as e:
            logger.warning("联网搜索失败，降级处理: {}", e)

    return local_docs, web_docs
