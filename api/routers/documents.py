"""文档管理 API 端点。"""

import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from api.schemas import (
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from config.settings import settings
from core.document_processor import process_document
from core.hybrid_retriever import rebuild_bm25_index
from core.vectorstore import add_documents, delete_by_document_id
from db.database import get_db
from db.models import Chunk, Document
from manifest.manager import check_file, remove_from_manifest, update_manifest

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".md", ".markdown", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 文件名级并发锁：防止同一文件同时上传导致数据竞争
_upload_locks: dict[str, threading.Lock] = {}
_upload_locks_guard = threading.Lock()


def _get_upload_lock(filename: str) -> threading.Lock:
    with _upload_locks_guard:
        if filename not in _upload_locks:
            _upload_locks[filename] = threading.Lock()
        return _upload_locks[filename]


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    raw_name = Path(file.filename or "").name
    lock = _get_upload_lock(raw_name)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"文件 {raw_name} 正在处理中，请稍后重试")
    try:
        return await _do_upload(file, raw_name, db)
    finally:
        lock.release()
        with _upload_locks_guard:
            _upload_locks.pop(raw_name, None)


async def _do_upload(file: UploadFile, raw_name: str, db: Session):
    try:
        # 1. 校验文件格式
        suffix = Path(raw_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {suffix}，仅支持 {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 2. 流式保存文件，边写边检查大小
        save_dir = Path(settings.UPLOAD_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / raw_name
        file_size = 0
        CHUNK_LIMIT = 1024 * 1024  # 1MB per read
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_LIMIT)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    f.close()
                    file_path.unlink()
                    raise HTTPException(status_code=400, detail="文件大小不能超过 50MB")
                f.write(chunk)

        if file_size == 0:
            file_path.unlink()
            raise HTTPException(status_code=400, detail="文件为空")

        # 4. Manifest 检查
        check = check_file(str(file_path), raw_name)
        if not check["changed"]:
            existing = db.query(Document).filter_by(filename=raw_name).first()
            return UploadResponse(
                document_id=existing.id if existing else "",
                filename=raw_name,
                chunk_count=existing.chunk_count if existing else 0,
                status="unchanged",
            )

        # 5. 清理旧数据（更新场景）
        old_doc = db.query(Document).filter_by(filename=raw_name).first()
        if old_doc:
            delete_by_document_id(old_doc.id)
            db.query(Chunk).filter_by(document_id=old_doc.id).delete()
            db.delete(old_doc)
            db.commit()
            document_id = old_doc.id
        else:
            document_id = str(uuid.uuid4())

        # 6. 处理文档：加载 → 分块
        chunks, _ = process_document(str(file_path), document_id=document_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空，无法索引")

        # 7. 存入 Chroma
        add_documents(chunks, document_id=document_id)

        # 8. 写入 SQLite
        doc = Document(
            id=document_id,
            filename=raw_name,
            file_path=str(file_path),
            file_size=file_size,
            file_type=suffix.lstrip("."),
            md5=check["md5"],
            chunk_count=len(chunks),
        )
        db.add(doc)
        for i, chunk in enumerate(chunks):
            db.add(Chunk(
                document_id=document_id,
                chunk_index=i,
                content=chunk.page_content,
                metadata_json=json.dumps(chunk.metadata, ensure_ascii=False),
            ))
        db.commit()

        # 9. 更新 Manifest
        update_manifest(raw_name, check["md5"], len(chunks))

        # 10. 重建 BM25 索引
        rebuild_bm25_index()

        status = "updated" if old_doc else "new"
        logger.info("文档上传成功: {} ({} chunks, {})", raw_name, len(chunks), status)
        return UploadResponse(
            document_id=document_id,
            filename=raw_name,
            chunk_count=len(chunks),
            status=status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("文档上传异常: {}", e)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {e}")


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.upload_time.desc()).all()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d.id, filename=d.filename, file_type=d.file_type,
                file_size=d.file_size, chunk_count=d.chunk_count,
                upload_time=d.upload_time,
            )
            for d in docs
        ]
    )


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(id=document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 1. 从 Chroma 删除（含验证）
    success = delete_by_document_id(document_id)
    if not success:
        logger.error("Chroma 删除失败，文档可能未完全清理: {}", doc.filename)

    # 2. 从 SQLite 删除
    db.query(Chunk).filter_by(document_id=document_id).delete()
    db.delete(doc)
    db.commit()

    # 3. 从 Manifest 删除
    remove_from_manifest(doc.filename)

    # 4. 删除上传文件
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()

    # 5. 重建 BM25 索引
    rebuild_bm25_index()

    logger.info("文档已删除: {}", doc.filename)
    return {"status": "deleted", "filename": doc.filename}
