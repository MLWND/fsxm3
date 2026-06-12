"""FastAPI 依赖注入。"""

from core.llm import get_llm
from core.retriever import retrieve_and_rerank
from db.database import get_db
