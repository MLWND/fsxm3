"""LLM 工厂：通过 OpenAI 兼容接口调用智谱 GLM。"""

from langchain_openai import ChatOpenAI

from config.settings import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.ZHIPUAI_MODEL,
        api_key=settings.ZHIPUAI_API_KEY,
        base_url=settings.ZHIPUAI_BASE_URL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
    )
