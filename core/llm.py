"""LLM 工厂：通过 OpenAI 兼容接口调用智谱 GLM，含重试机制。"""

import time

from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import settings

# 重试配置
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.ZHIPUAI_MODEL,
        api_key=settings.ZHIPUAI_API_KEY,
        base_url=settings.ZHIPUAI_BASE_URL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )


def invoke_with_retry(messages, **kwargs):
    """调用 LLM 并带指数退避重试，处理 429/5xx 错误。"""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            llm = get_llm()
            return llm.invoke(messages, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # 判断是否可重试（429 rate limit 或 5xx 服务端错误）
            is_retryable = any(
                code in error_str
                for code in ("429", "rate limit", "500", "502", "503", "504")
            )
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise

            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "LLM 调用失败 (attempt {}/{}), {}s 后重试: {}",
                attempt + 1, MAX_RETRIES, wait, e,
            )
            time.sleep(wait)

    raise last_error  # type: ignore[misc]
