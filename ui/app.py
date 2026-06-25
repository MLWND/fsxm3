"""Streamlit 前端：智能知识库问答界面（支持流式输出）。"""

import hashlib
import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="智能知识库问答", layout="wide")

# ====== Material Icons + 自定义样式 ======
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
<style>
    .icon { font-family: 'Material Icons Outlined'; font-size: 18px; vertical-align: middle; margin-right: 4px; }
    .icon-sm { font-family: 'Material Icons Outlined'; font-size: 15px; vertical-align: middle; }
    .stChatMessage { padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ====== Session State 初始化 ======
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()
if "pending_sources" not in st.session_state:
    st.session_state.pending_sources = None


# ====== 工具函数 ======
def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("无法连接后端服务，请确认 FastAPI 已启动 (localhost:8000)")
        return None
    except requests.RequestException as e:
        st.error(f"API 错误: {e}")
        return None


def api_post(path: str, json_data=None, files=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_data, files=files, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("无法连接后端服务，请确认 FastAPI 已启动 (localhost:8000)")
        return None
    except requests.RequestException as e:
        st.error(f"API 错误: {e}")
        return None


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API 错误: {e}")
        return None


def _sse_stream(path: str, json_data: dict):
    """调用 SSE 端点，yield 每个 token。结束后设置 st.session_state.pending_sources。"""
    try:
        with requests.post(f"{API_BASE}{path}", json=json_data, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "token":
                    yield event["content"]
                elif event.get("type") == "done":
                    st.session_state.pending_sources = event.get("sources", [])
                    if event.get("conversation_id"):
                        st.session_state.conversation_id = event["conversation_id"]
                elif event.get("type") == "error":
                    st.error(f"LLM 错误: {event.get('message', '未知错误')}")
    except requests.ConnectionError:
        st.error("无法连接后端服务，请确认 FastAPI 已启动 (localhost:8000)")
    except requests.RequestException as e:
        st.error(f"API 错误: {e}")


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def new_conversation():
    st.session_state.conversation_id = None
    st.session_state.messages = []


# ====== 侧边栏 ======
with st.sidebar:
    st.markdown("## <span class='icon'>menu_book</span> 知识库问答系统", unsafe_allow_html=True)

    # --- 文档上传 ---
    st.markdown("### <span class='icon'>upload_file</span> 上传文档", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "拖拽或点击上传",
        type=["txt", "pdf", "docx", "csv", "md", "markdown", "xlsx"],
        label_visibility="collapsed",
        help="支持 TXT、PDF、DOCX、CSV、Markdown、Excel，最大 50MB",
    )
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash not in st.session_state.uploaded_hashes:
            with st.spinner(f"正在处理 {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, file_bytes)}
                result = api_post("/documents/upload", files=files)
                if result:
                    st.session_state.uploaded_hashes.add(file_hash)
                    if result["status"] == "unchanged":
                        st.info(f"{result['filename']} 未变化，已跳过")
                    else:
                        st.success(f"{result['filename']} ({result['chunk_count']} 个文本片段)")

    # --- 文档列表 ---
    st.markdown("### <span class='icon'>folder_open</span> 已上传文档", unsafe_allow_html=True)
    docs_data = api_get("/documents")
    if docs_data and docs_data["documents"]:
        total_chunks = 0
        for doc in docs_data["documents"]:
            total_chunks += doc["chunk_count"]
            col1, col2 = st.columns([5, 1])
            with col1:
                st.caption(f"**{doc['filename']}**")
                st.caption(f"{doc['chunk_count']} 片段 · {format_file_size(doc['file_size'])}")
            with col2:
                if st.button("×", key=f"del_{doc['id']}", help="删除文档"):
                    api_delete(f"/documents/{doc['id']}")
                    st.rerun()
        st.caption(f"共 {len(docs_data['documents'])} 个文档，{total_chunks} 个片段")
    else:
        st.info("暂无文档，请上传后开始问答")

    st.divider()

    # --- 功能开关 ---
    st.markdown("### <span class='icon'>tune</span> 功能", unsafe_allow_html=True)
    use_rewrite = st.checkbox(
        "查询改写",
        value=False,
        help="启用后，LLM 会将你的口语化问题改写为更精确的检索查询",
    )
    use_hybrid = st.checkbox(
        "混合检索",
        value=False,
        help="BM25 关键词 + 向量语义，双重检索融合，效果更好",
    )
    use_stream = st.checkbox(
        "流式输出",
        value=True,
        help="逐字显示回答，无需等待全部生成",
    )

    st.divider()

    # --- 对话管理 ---
    st.markdown("### <span class='icon'>chat</span> 对话", unsafe_allow_html=True)
    if st.button("+ 新建对话", use_container_width=True):
        new_conversation()
        st.rerun()

    convs_data = api_get("/conversations")
    if convs_data and convs_data["conversations"]:
        for conv in convs_data["conversations"]:
            is_active = conv["id"] == st.session_state.conversation_id
            label = f"{'* ' if is_active else ''}{conv['title'][:25]}"
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state.conversation_id = conv["id"]
                    history = api_get(f"/history/{conv['id']}")
                    if history:
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in history["messages"]
                        ]
                    st.rerun()
            with col2:
                if st.button("×", key=f"del_conv_{conv['id']}", help="删除对话"):
                    api_delete(f"/conversations/{conv['id']}")
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                    st.rerun()


# ====== 主聊天区域 ======
st.markdown("## <span class='icon'>question_answer</span> 向知识库提问", unsafe_allow_html=True)

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 空状态提示
if not st.session_state.messages:
    st.markdown("""
    欢迎使用智能知识库问答系统！使用步骤：

    1. 在左侧上传文档（TXT / PDF / DOCX / CSV / Markdown）
    2. 在下方输入问题
    3. 系统将基于文档内容为你回答
    """)

# 用户输入
if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_sources = None
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_data = {
        "message": prompt,
        "conversation_id": st.session_state.conversation_id,
        "use_rewrite": use_rewrite,
        "use_hybrid": use_hybrid,
    }

    with st.chat_message("assistant"):
        if use_stream:
            # 流式输出：逐字显示
            answer = st.write_stream(
                _sse_stream("/chat/stream", chat_data)
            )
        else:
            # 非流式输出
            with st.spinner("正在检索知识库并生成回答..."):
                result = api_post("/chat", json_data=chat_data)
            if result:
                answer = result["answer"]
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.pending_sources = result.get("sources", [])
                st.markdown(answer)
            else:
                answer = ""

        # 来源展示
        if st.session_state.pending_sources:
            with st.expander("参考来源", expanded=False):
                for i, src in enumerate(st.session_state.pending_sources, 1):
                    page_info = f" · 第 {src['page'] + 1} 页" if src["page"] >= 0 else ""
                    st.markdown(f"**{i}. {src['filename']}**{page_info}（片段 {src['chunk_index'] + 1}）")
                    st.caption(src["snippet"])

    # 保存助手消息
    if answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
