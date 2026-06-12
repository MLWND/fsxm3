"""Embedding 工厂：本地 HuggingFace BGE 模型。"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )
