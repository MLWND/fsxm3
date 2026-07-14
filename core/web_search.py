"""Exa 联网搜索：封装 Exa Search REST API，返回 LangChain Document 列表。"""

import requests
from langchain_core.documents import Document
from loguru import logger

from config.settings import settings

EXA_SEARCH_URL = "https://api.exa.ai/search"


def web_search(query: str, top_k: int = 5) -> list[Document]:
    """调用 Exa Search API，返回 LangChain Document 列表。

    Args:
        query: 搜索查询（自然语言）
        top_k: 返回结果数，默认 5

    Returns:
        Document 列表，metadata 包含 source_type / source_url / title / published_date

    Raises:
        ValueError: EXA_API_KEY 未配置
        RuntimeError: API 调用失败
    """
    if not settings.EXA_API_KEY:
        raise ValueError("EXA_API_KEY 未配置，请在 .env 中设置后重启服务")

    payload = {
        "query": query,
        "numResults": top_k,
        "contents": {"highlights": True},
        "type": "auto",
    }

    try:
        resp = requests.post(
            EXA_SEARCH_URL,
            json=payload,
            headers={
                "x-api-key": settings.EXA_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("Exa 搜索失败: {}", e)
        raise RuntimeError(f"Exa 搜索失败: {e}") from e

    results = data.get("results", [])
    docs = []
    for r in results:
        highlights = r.get("highlights", [])
        content = "\n".join(highlights) if highlights else ""
        docs.append(Document(
            page_content=content,
            metadata={
                "source_type": "web",
                "source_url": r.get("url", ""),
                "title": r.get("title", ""),
                "published_date": r.get("publishedDate") or "",
            },
        ))

    logger.info("Exa 搜索完成: query='{}' → {} 条结果", query[:80], len(docs))
    return docs
