"""查询改写：LLM 将口语化问题改写为更适合检索的查询。"""

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from core.llm import get_llm
from prompts.templates import QUERY_REWRITE_TEMPLATE


def rewrite_query(question: str) -> str:
    """将用户原始问题改写为更适合向量检索的查询。

    例如:
        "LC是啥" → "LangChain是什么框架"
        "那个切块的东西" → "文本分块器 TextSplitter"
    """
    try:
        llm = get_llm()
        prompt = QUERY_REWRITE_TEMPLATE.format(question=question)
        response = llm.invoke([
            SystemMessage(content="你是查询优化专家，只输出改写后的查询，不要解释。"),
            HumanMessage(content=prompt),
        ])
        rewritten = response.content.strip()
        logger.debug("查询改写: '{}' → '{}'", question, rewritten)
        return rewritten
    except Exception as e:
        logger.warning("查询改写失败，使用原始问题: {}", e)
        return question
