"""文档处理：加载 TXT/PDF/DOCX → 分块 → 带 metadata。"""

import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config.settings import settings


def _load_txt(file_path: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": file_path})]


def _load_pdf(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(file_path).load()


def _load_docx(file_path: str) -> list[Document]:
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    text = "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )
    return [Document(page_content=text, metadata={"source": file_path})]


LOADERS = {
    ".txt": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )


def process_document(
    file_path: str,
    document_id: str | None = None,
) -> tuple[list[Document], str]:
    """加载文档、分块、附加 metadata。

    Returns:
        (chunks, document_id)
    """
    file_path = str(Path(file_path).resolve())
    suffix = Path(file_path).suffix.lower()

    if suffix not in LOADERS:
        raise ValueError(f"不支持的文件格式: {suffix}")

    document_id = document_id or str(uuid.uuid4())
    filename = Path(file_path).name

    logger.info("加载文档: {} (format={})", filename, suffix)
    raw_docs = LOADERS[suffix](file_path)
    logger.info("原始文档: {} 页/段", len(raw_docs))

    splitter = get_text_splitter()
    chunks = splitter.split_documents(raw_docs)
    logger.info("分块完成: {} chunks", len(chunks))

    now = datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        # 保留原始 page 信息（PDF 的 PyPDFLoader 自带）
        page = chunk.metadata.get("page", chunk.metadata.get("page_number", -1))

        chunk.metadata.update({
            "document_id": document_id,
            "source": filename,
            "page": page,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "upload_time": now,
        })

    return chunks, document_id
