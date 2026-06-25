"""对话 API 端点：支持普通问答 + 流式输出。"""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from sqlalchemy.orm import Session

from api.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationSummary,
    HistoryResponse,
    SourceInfo,
)
from config.settings import settings
from core.llm import get_llm, invoke_with_retry
from core.query_rewrite import rewrite_query
from core.retriever import retrieve
from db.database import get_db
from db.models import Conversation, Message
from prompts.templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

router = APIRouter()


# ====== 内部工具函数 ======

def _get_or_create_conversation(req: ChatRequest, db: Session) -> tuple[Conversation, bool]:
    """获取或创建对话记录。返回 (conversation, is_new)。"""
    is_new = False
    if req.conversation_id:
        conv = db.query(Conversation).filter_by(id=req.conversation_id).first()
        if not conv:
            conv = Conversation(id=req.conversation_id, title=req.message[:50])
            db.add(conv)
            is_new = True
    else:
        conv = Conversation(title=req.message[:50])
        db.add(conv)
        is_new = True
    db.commit()
    return conv, is_new


def _auto_title(conv: Conversation, message: str, db: Session):
    """用 LLM 为新对话生成简短标题，失败则截断原文。"""
    try:
        from core.llm import invoke_with_retry
        from langchain_core.messages import HumanMessage, SystemMessage

        response = invoke_with_retry([
            SystemMessage(content="你是标题生成专家。根据用户的第一条消息，生成一个不超过15个字的简短标题。只输出标题，不要引号或解释。"),
            HumanMessage(content=message),
        ])
        title = response.content.strip().strip('"').strip("'")[:50]
        if title:
            conv.title = title
            db.commit()
    except Exception:
        conv.title = message[:50]
        db.commit()


def _auto_title_async(conv_id: str, message: str):
    """后台线程生成标题，不阻塞主流程。"""
    import threading
    from db.database import SessionLocal

    def _worker():
        try:
            _db = SessionLocal()
            _conv = _db.query(Conversation).filter_by(id=conv_id).first()
            if _conv:
                _auto_title(_conv, message, _db)
            _db.close()
        except Exception as e:
            logger.warning("标题生成失败: {}", e)

    threading.Thread(target=_worker, daemon=True).start()


def _build_history_block(conversation_id: str, db: Session) -> str:
    """构建对话历史块，基于 token 上限窗口裁剪。"""
    history_msgs = (
        db.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at)
        .all()
    )
    if len(history_msgs) <= 1:
        return ""

    # 排除最后一条（刚保存的 user_msg），从后往前累积直到 token 上限
    past = history_msgs[:-1]
    max_chars = settings.HISTORY_MAX_TOKENS * 2  # 粗估 1 token ≈ 2 字符（中文）
    lines = []
    char_count = 0
    for m in reversed(past):
        role_label = "用户" if m.role == "user" else "助手"
        line = f"{role_label}：{m.content}"
        line_len = len(line)
        if char_count + line_len > max_chars:
            break
        lines.append(line)
        char_count += line_len
    lines.reverse()
    return "\n".join(lines)


def _build_rag_context(docs, req_message: str, conversation_id: str, db: Session):
    """组装 RAG 上下文、来源和完整 prompt。返回 (answer_or_none, sources, messages_for_llm)。"""
    if not docs:
        return "知识库中暂无相关内容，请先上传文档后再提问。", [], None

    context_parts = []
    sources = []
    seen = set()
    for d in docs:
        context_parts.append(d.page_content)
        filename = d.metadata.get("source", "未知来源")
        key = f"{filename}_{d.metadata.get('chunk_index', 0)}"
        if key not in seen:
            seen.add(key)
            snippet = d.page_content[:150] + ("..." if len(d.page_content) > 150 else "")
            sources.append(SourceInfo(
                filename=filename,
                page=d.metadata.get("page", -1),
                chunk_index=d.metadata.get("chunk_index", 0),
                snippet=snippet,
            ))
    context = "\n\n---\n\n".join(context_parts)

    # 构建消息列表
    history_block = _build_history_block(conversation_id, db)
    user_prompt = RAG_USER_TEMPLATE.format(context=context, question=req_message)
    if history_block:
        user_prompt = f"## 对话历史\n{history_block}\n\n## 当前问题\n{user_prompt}"

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    return None, sources, messages


# ====== 普通问答（非流式）======

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    try:
        conv, is_new = _get_or_create_conversation(req, db)

        # 保存用户消息
        user_msg = Message(conversation_id=conv.id, role="user", content=req.message)
        db.add(user_msg)
        db.commit()

        # 新对话首条消息：自动生成标题
        if is_new:
            _auto_title(conv, req.message, db)

        # 查询改写 + 检索
        search_query = rewrite_query(req.message) if req.use_rewrite else req.message
        docs = retrieve(search_query, use_hybrid=req.use_hybrid)

        answer_or_none, sources, messages = _build_rag_context(docs, req.message, conv.id, db)
        if answer_or_none:
            answer = answer_or_none
        else:
            response = invoke_with_retry(messages)
            answer = response.content

        # 保存助手消息
        assistant_msg = Message(conversation_id=conv.id, role="assistant", content=answer)
        db.add(assistant_msg)
        conv.updated_at = datetime.now()
        db.commit()

        return ChatResponse(
            conversation_id=conv.id,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        logger.error("聊天接口异常: {}", e)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# ====== 流式问答（SSE）======

@router.post("/chat/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """SSE 流式输出：逐 token 返回回答 + 结束时返回完整 sources JSON。"""
    try:
        conv, is_new = _get_or_create_conversation(req, db)

        # 保存用户消息
        user_msg = Message(conversation_id=conv.id, role="user", content=req.message)
        db.add(user_msg)
        db.commit()

        # 新对话首条消息：后台生成标题（不阻塞流式输出）
        if is_new:
            _auto_title_async(conv.id, req.message)

        # 查询改写 + 检索
        search_query = rewrite_query(req.message) if req.use_rewrite else req.message
        docs = retrieve(search_query, use_hybrid=req.use_hybrid)

        answer_or_none, sources, messages = _build_rag_context(docs, req.message, conv.id, db)
        if answer_or_none:
            # 无检索结果，直接返回
            def empty_gen():
                yield f"data: {json.dumps({'type': 'token', 'content': answer_or_none})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': [s.model_dump() for s in sources]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_gen(), media_type="text/event-stream")

        llm = get_llm()

        def stream_gen():
            full_answer = []
            try:
                for chunk in llm.stream(messages):
                    token = chunk.content
                    if token:
                        full_answer.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # 流式结束后保存完整回答到数据库
                answer_text = "".join(full_answer)
                if answer_text:
                    from db.database import SessionLocal
                    save_db = SessionLocal()
                    try:
                        save_db.add(Message(
                            conversation_id=conv.id, role="assistant", content=answer_text,
                        ))
                        conv.updated_at = datetime.now()
                        save_db.commit()
                    finally:
                        save_db.close()

                yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv.id, 'sources': [s.model_dump() for s in sources]})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error("流式生成异常: {}", e)
                yield f"data: {json.dumps({'type': 'error', 'message': '生成过程中发生错误'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    except Exception as e:
        logger.error("聊天接口异常: {}", e)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# ====== 对话管理 ======

@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(db: Session = Depends(get_db)):
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return ConversationListResponse(
        conversations=[
            ConversationSummary(
                id=c.id, title=c.title,
                created_at=c.created_at, updated_at=c.updated_at,
            )
            for c in convs
        ]
    )


@router.get("/history/{conversation_id}", response_model=HistoryResponse)
def get_history(conversation_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at)
        .all()
    )
    return HistoryResponse(
        messages=[
            ChatMessageResponse(
                id=m.id, role=m.role, content=m.content, created_at=m.created_at,
            )
            for m in messages
        ]
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    db.query(Message).filter_by(conversation_id=conversation_id).delete()
    db.delete(conv)
    db.commit()
    return {"status": "deleted"}
