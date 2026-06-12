"""检索 pipeline：MMR 或 Hybrid Search + 可选 Rerank。"""

from langchain_core.documents import Document

from config.settings import settings
from core.vectorstore import get_vectorstore


def retrieve(query: str, top_k: int | None = None, use_hybrid: bool = False) -> list[Document]:
    """检索入口。

    Args:
        query: 搜索查询
        top_k: 返回结果数
        use_hybrid: 是否使用混合检索（BM25 + Vector）
    """
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

    # Rerank 精排
    from core.reranker import get_reranker
    model = get_reranker()
    if model is None:
        return candidates[:top_k]

    from core.reranker import rerank
    return rerank(query, candidates, top_k=top_k)
