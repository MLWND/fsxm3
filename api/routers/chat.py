"""对话 API 端点。"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
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
from core.llm import get_llm
from core.query_rewrite import rewrite_query
from core.retriever import retrieve
from db.database import get_db
from db.models import Conversation, Message
from prompts.templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    try:
        # 1. 获取或创建对话
        if req.conversation_id:
            conv = db.query(Conversation).filter_by(id=req.conversation_id).first()
            if not conv:
                conv = Conversation(id=req.conversation_id, title=req.message[:50])
                db.add(conv)
        else:
            conv = Conversation(title=req.message[:50])
            db.add(conv)
        db.commit()

        # 2. 保存用户消息
        user_msg = Message(conversation_id=conv.id, role="user", content=req.message)
        db.add(user_msg)
        db.commit()

        # 3. 查询改写（可选）+ 检索
        search_query = rewrite_query(req.message) if req.use_rewrite else req.message
        docs = retrieve(search_query, use_hybrid=req.use_hybrid)

        # 4. 没有检索到内容时
        if not docs:
            answer = "知识库中暂无相关内容，请先上传文档后再提问。"
            sources = []
        else:
            # 5. 组装上下文和来源（含片段）
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

            # 6. LLM 生成回答
            llm = get_llm()
            user_prompt = RAG_USER_TEMPLATE.format(context=context, question=req.message)

            response = llm.invoke([
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            answer = response.content

        # 7. 保存助手消息
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
    db.query(Message).filter_by(conversation_id=conversation_id).delete()
    db.query(Conversation).filter_by(id=conversation_id).delete()
    db.commit()
    return {"status": "deleted"}
