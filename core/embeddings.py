"""Embedding 工厂：本地 HuggingFace BGE 模型，自动检测 GPU。"""

import torch
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


def _get_device() -> str:
    """自动检测可用设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    device = _get_device()
    import os
    # 优先使用本地模型路径（Docker 预下载），否则用配置的模型名
    local_model = "/app/models/bge-small-zh-v1.5"
    model_name = local_model if os.path.isdir(local_model) else settings.EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32 if device == "cpu" else 64,
        },
    )
