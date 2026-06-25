"""混合检索：BM25 (关键词) + Chroma (语义) → EnsembleRetriever。

效果：兼顾精确关键词匹配和语义理解，比单一检索质量更高。
"""

import hashlib

import jieba
from langchain_core.documents import Document
from loguru import logger
from rank_bm25 import BM25Okapi

from config.settings import settings
from core.vectorstore import get_vectorstore


def _tokenize(text: str) -> list[str]:
    """中文分词：jieba 切词 + 小写化。"""
    return [w.lower() for w in jieba.lcut(text) if w.strip()]


def _doc_key(doc: Document) -> str:
    """生成文档唯一标识：source + chunk_index + 内容哈希，避免前缀碰撞。"""
    src = doc.metadata.get("source", "")
    idx = doc.metadata.get("chunk_index", "")
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
    return f"{src}:{idx}:{content_hash}"


class HybridRetriever:
    """BM25 + 向量混合检索器。"""

    def __init__(self):
        self._bm25_docs: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def build_bm25_index(self, documents: list[Document]):
        """从文档列表构建 BM25 索引。"""
        if not documents:
            logger.warning("空文档列表，跳过 BM25 索引构建")
            return

        self._bm25_docs = documents
        corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 索引已构建: {} 篇文档", len(documents))

    def search_bm25(self, query: str, top_k: int | None = None) -> list[Document]:
        """BM25 关键词检索。"""
        if self._bm25 is None or not self._bm25_docs:
            return []

        top_k = top_k or settings.RETRIEVAL_TOP_K
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 取 top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score > 0:
                results.append(self._bm25_docs[idx])
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int | None = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> list[Document]:
        """混合检索：BM25 分数 × 权重 + 向量相似度 × 权重。"""
        top_k = top_k or settings.RERANK_TOP_K

        # BM25 结果
        bm25_docs = self.search_bm25(query, top_k=settings.RETRIEVAL_TOP_K)
        # 向量结果
        vs = get_vectorstore()
        vector_docs = vs.max_marginal_relevance_search(
            query,
            k=settings.RETRIEVAL_TOP_K,
            fetch_k=settings.RETRIEVAL_TOP_K,
            lambda_mult=0.7,
        )

        # RRF 融合 (Reciprocal Rank Fusion)
        # 比加权平均更稳定，不需要归一化分数
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = _doc_key(doc)
            doc_scores[key] = doc_scores.get(key, 0) + bm25_weight / (60 + rank)
            doc_map[key] = doc

        for rank, doc in enumerate(vector_docs):
            key = _doc_key(doc)
            doc_scores[key] = doc_scores.get(key, 0) + vector_weight / (60 + rank)
            doc_map[key] = doc

        # 按融合分数排序
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [doc_map[key] for key, _ in ranked]

        logger.debug(
            "混合检索: BM25={} 条, Vector={} 条 → 融合={} 条",
            len(bm25_docs), len(vector_docs), len(results),
        )
        return results


# ====== 全局实例 + BM25 索引管理 ======

_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def rebuild_bm25_index():
    """从 Chroma 中的所有文档重建 BM25 索引。"""
    try:
        vs = get_vectorstore()
        collection = vs._collection
        if collection.count() == 0:
            logger.info("Chroma 为空，跳过 BM25 索引构建")
            return

        # 获取所有文档
        results = collection.get(include=["documents", "metadatas"])
        docs = []
        for content, metadata in zip(results["documents"], results["metadatas"], strict=False):
            docs.append(Document(page_content=content, metadata=metadata or {}))

        retriever = get_hybrid_retriever()
        retriever.build_bm25_index(docs)
    except Exception as e:
        logger.error("BM25 索引构建失败: {}", e)
