"""集中化配置管理，通过 .env 文件加载。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
    )

    # === 智谱 AI（OpenAI 兼容接口）===
    ZHIPUAI_API_KEY: str = ""
    ZHIPUAI_MODEL: str = "glm-4.7-flash"
    ZHIPUAI_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"

    # === Embedding ===
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # === Reranker（V2 可选）===
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # === Chroma ===
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    CHROMA_COLLECTION: str = "knowledge_base"

    # === 分块 ===
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # === 检索 ===
    RETRIEVAL_TOP_K: int = 20  # 召回多一些，给 Reranker
    RERANK_TOP_K: int = 5  # Rerank 后取 top 5

    # === 路径 ===
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    SQLITE_DB: str = str(BASE_DIR / "data" / "knowledge_base.db")
    MANIFEST_PATH: str = str(BASE_DIR / "data" / "manifest.json")
    LOG_DIR: str = str(BASE_DIR / "logs")

    # === LLM 生成参数 ===
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2048

    # === API ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
