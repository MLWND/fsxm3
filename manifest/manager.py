"""Manifest 增量同步管理：MD5 比对，避免重复 Embedding。"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import settings


def _compute_md5(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    path = Path(settings.MANIFEST_PATH)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(data: dict) -> None:
    path = Path(settings.MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_file(file_path: str, filename: str) -> dict:
    """检查文件是否需要重新处理。

    Returns:
        {"changed": bool, "md5": str, "existing_entry": dict | None}
    """
    md5 = _compute_md5(file_path)
    manifest = _load_manifest()

    if filename in manifest and manifest[filename]["md5"] == md5:
        logger.info("文件未变化，跳过: {}", filename)
        return {"changed": False, "md5": md5, "existing_entry": manifest[filename]}

    return {"changed": True, "md5": md5, "existing_entry": manifest.get(filename)}


def update_manifest(filename: str, md5: str, chunk_count: int) -> None:
    """上传/更新后写入 manifest。"""
    manifest = _load_manifest()
    manifest[filename] = {
        "md5": md5,
        "chunk_count": chunk_count,
        "last_update": datetime.now().isoformat(),
    }
    _save_manifest(manifest)
    logger.info("Manifest 已更新: {} ({} chunks)", filename, chunk_count)


def remove_from_manifest(filename: str) -> int:
    """删除文档条目，返回 chunk_count 用于清理 Chroma。"""
    manifest = _load_manifest()
    entry = manifest.pop(filename, {"chunk_count": 0})
    _save_manifest(manifest)
    logger.info("Manifest 已删除: {}", filename)
    return entry.get("chunk_count", 0)


def list_manifest() -> dict:
    return _load_manifest()
