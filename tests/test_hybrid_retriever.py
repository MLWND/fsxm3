"""测试混合检索：RRF key 唯一性 + BM25 构建。"""

from langchain_core.documents import Document

from core.hybrid_retriever import HybridRetriever, _doc_key


class TestDocKey:
    def test_unique_for_different_content(self):
        """不同内容应生成不同的 key。"""
        d1 = Document(page_content="内容A" * 20, metadata={"source": "a.txt", "chunk_index": 0})
        d2 = Document(page_content="内容B" * 20, metadata={"source": "a.txt", "chunk_index": 0})
        assert _doc_key(d1) != _doc_key(d2)

    def test_unique_for_same_prefix(self):
        """前 100 字符相同但整体不同，key 也应不同。"""
        prefix = "这是一段很长的文本前缀，用来测试碰撞问题。" * 3
        d1 = Document(page_content=prefix + "结尾A", metadata={"source": "a.txt", "chunk_index": 0})
        d2 = Document(page_content=prefix + "结尾B", metadata={"source": "a.txt", "chunk_index": 0})
        assert _doc_key(d1) != _doc_key(d2)

    def test_same_content_same_key(self):
        """相同内容 + 相同 metadata 应生成相同 key。"""
        d1 = Document(page_content="完全一样的内容", metadata={"source": "a.txt", "chunk_index": 0})
        d2 = Document(page_content="完全一样的内容", metadata={"source": "a.txt", "chunk_index": 0})
        assert _doc_key(d1) == _doc_key(d2)


class TestBM25:
    def test_build_and_search(self):
        """构建 BM25 索引后应能检索到结果。"""
        retriever = HybridRetriever()
        docs = [
            Document(page_content="LangChain 是一个 LLM 应用框架", metadata={"source": "a.txt"}),
            Document(page_content="FastAPI 是一个高性能 Web 框架", metadata={"source": "b.txt"}),
            Document(page_content="PyTorch 是深度学习框架", metadata={"source": "c.txt"}),
        ]
        retriever.build_bm25_index(docs)
        results = retriever.search_bm25("LangChain 框架", top_k=2)
        assert len(results) >= 1
        assert any("LangChain" in d.page_content for d in results)

    def test_empty_index_returns_empty(self):
        """空索引检索应返回空列表。"""
        retriever = HybridRetriever()
        results = retriever.search_bm25("test")
        assert results == []
