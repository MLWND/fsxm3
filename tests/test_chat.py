"""测试对话 API：delete 存在性检查 + 健康检查。"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """创建测试客户端（mock 外部依赖）。"""
    with patch("core.embeddings.get_embeddings", return_value=MagicMock()):
        with patch("core.vectorstore.get_vectorstore", return_value=MagicMock(count=MagicMock(return_value=0))):
            from api.main import app
            yield TestClient(app)


class TestDeleteConversation:
    def test_delete_nonexistent_returns_404(self, client):
        """删除不存在的对话应返回 404。"""
        resp = client.delete("/api/conversations/nonexistent-id")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_delete_existing_returns_200(self, client):
        """删除存在的对话应返回 200。"""
        # Mock 检索和 LLM，避免真实网络调用
        with patch("api.routers.chat.retrieve", return_value=([], [])):
            with patch("api.routers.chat.rewrite_query", side_effect=lambda q: q):
                resp = client.post("/api/chat", json={"message": "测试消息"})
                assert resp.status_code == 200
                conv_id = resp.json()["conversation_id"]

        # 删除
        resp = client.delete(f"/api/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # 验证确实删了
        resp = client.delete(f"/api/conversations/{conv_id}")
        assert resp.status_code == 404


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """健康检查端点应返回 200。"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "documents" in data
        assert "chunks" in data


class TestSearchMode:
    """测试 search_mode 参数。"""

    def test_chat_request_default_search_mode(self):
        """ChatRequest 默认 search_mode 为 local。"""
        from api.schemas import ChatRequest
        req = ChatRequest(message="hello")
        assert req.search_mode == "local"

    def test_chat_request_web_mode(self):
        """ChatRequest search_mode 可设为 web。"""
        from api.schemas import ChatRequest
        req = ChatRequest(message="hello", search_mode="web")
        assert req.search_mode == "web"

    def test_chat_request_hybrid_mode(self):
        """ChatRequest search_mode 可设为 hybrid。"""
        from api.schemas import ChatRequest
        req = ChatRequest(message="hello", search_mode="hybrid")
        assert req.search_mode == "hybrid"


class TestWebSourceInfo:
    """测试 WebSourceInfo 模型。"""

    def test_web_source_info(self):
        """WebSourceInfo 序列化正确。"""
        from api.schemas import WebSourceInfo
        w = WebSourceInfo(title="Test", url="https://example.com", snippet="snippet text")
        d = w.model_dump()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"
        assert d["snippet"] == "snippet text"
