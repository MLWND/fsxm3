"""web_search 模块测试。"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def test_web_search_returns_documents():
    """正常返回：Exa 响应转为 LangChain Document 列表。"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Test Page",
                "url": "https://example.com",
                "highlights": ["highlight 1", "highlight 2"],
                "publishedDate": "2024-01-01",
            },
            {
                "title": "Another Page",
                "url": "https://example2.com",
                "highlights": ["another highlight"],
                "publishedDate": None,
            },
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response) as mock_post:
        from core.web_search import web_search
        docs = web_search("test query", top_k=3)

    assert len(docs) == 2
    assert docs[0].page_content == "highlight 1\nhighlight 2"
    assert docs[0].metadata["source_type"] == "web"
    assert docs[0].metadata["source_url"] == "https://example.com"
    assert docs[0].metadata["title"] == "Test Page"
    assert docs[0].metadata["published_date"] == "2024-01-01"

    assert docs[1].metadata["published_date"] == ""
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.exa.ai/search"
    assert call_args[1]["json"]["query"] == "test query"
    assert call_args[1]["json"]["numResults"] == 3


def test_web_search_no_api_key():
    """EXA_API_KEY 为空时抛出 ValueError。"""
    with patch("core.web_search.settings") as mock_settings:
        mock_settings.EXA_API_KEY = ""
        from core.web_search import web_search
        with pytest.raises(ValueError, match="EXA_API_KEY"):
            web_search("query")


def test_web_search_api_error():
    """API 返回错误时抛出 RuntimeError。"""
    import requests as r
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = r.HTTPError("500 Server Error")

    with patch("requests.post", return_value=mock_response):
        from core.web_search import web_search
        with pytest.raises(RuntimeError, match="Exa 搜索失败"):
            web_search("query")


def test_web_search_empty_results():
    """无搜索结果时返回空列表。"""
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        from core.web_search import web_search
        docs = web_search("very obscure query")

    assert docs == []
