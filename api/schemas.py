"""Pydantic 请求/响应模型。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(max_length=10000)
    conversation_id: str | None = None
    use_rewrite: bool = False
    use_hybrid: bool = False
    search_mode: Literal["local", "web", "hybrid"] = "local"


class SourceInfo(BaseModel):
    filename: str
    page: int = -1         # 页码（TXT/DOCX 为 -1）
    chunk_index: int = 0   # 片段索引
    snippet: str           # 来源片段内容


class WebSourceInfo(BaseModel):
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceInfo | WebSourceInfo]


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime


class HistoryResponse(BaseModel):
    messages: list[ChatMessageResponse]


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str  # "new" / "updated" / "unchanged"


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    upload_time: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class ErrorResponse(BaseModel):
    """统一错误响应格式。"""
    detail: str
    error_code: str = "UNKNOWN"
    request_id: str | None = None
