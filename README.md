# 智能知识库问答系统

基于 LangChain + RAG 技术的智能知识库问答系统。支持多格式文档上传、增量同步、混合检索、Rerank 精排、查询改写和来源追溯。

## 核心流程

```
文档上传
  -> 格式校验（TXT/PDF/DOCX）
  -> MD5 Manifest 增量检测（跳过未变化文件）
  -> 中文分块（RecursiveCharacterTextSplitter + 中文分隔符）
  -> Embedding 向量化（bge-small-zh-v1.5，本地推理）
  -> 存入 Chroma 向量库 + SQLite 元数据 + Manifest 清单

用户提问
  -> [可选] 查询改写（LLM 将口语化问题改写为精确检索查询）
  -> 向量检索 top-20（MMR 或 BM25+Vector 混合检索）
  -> RRF 融合排序（混合检索模式下）
  -> Rerank 精排（bge-reranker-v2-m3 CrossEncoder）-> top-5
  -> 组装上下文 + Prompt 模板
  -> GLM-4.6V-Flash 生成回答
  -> 返回答案 + 来源文件 + 页码 + 内容片段预览
```

## 技术栈

| 模块 | 选择 | 说明 |
|------|------|------|
| LLM | GLM-4.6V-Flash (智谱 AI) | OpenAI 兼容接口调用 |
| RAG 框架 | LangChain | 文档加载、分块、检索、链式调用 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地推理，512 维，中文优化 |
| Reranker | BAAI/bge-reranker-v2-m3 | CrossEncoder 精排，2.2GB |
| 向量库 | Chroma | 持久化存储，支持 MMR 检索 |
| 混合检索 | BM25 (rank_bm25) + Chroma | jieba 中文分词，RRF 融合 |
| 前端 | Streamlit | Material Icons，响应式界面 |
| 后端 | FastAPI | REST API，Swagger 文档 |
| 数据库 | SQLite + SQLAlchemy ORM | 4 表设计，持久化对话历史 |
| 配置 | pydantic-settings | 类型安全，自动加载 .env |
| 日志 | loguru | 控制台 + 文件轮转（10MB/7天） |

## 功能特性

### 文档处理

- **多格式支持** — TXT、PDF、DOCX 三种格式
- **中文分块** — RecursiveCharacterTextSplitter，中文分隔符优先（`。！？；`），chunk_size=500
- **元数据丰富** — 每个 chunk 记录：文件名、页码（PDF）、片段索引、总片段数、上传时间
- **Manifest 增量同步** — MD5 比对，文件未变化时跳过 Embedding，避免重复计算
- **精确删除** — 删除文档时：Chroma 向量 + SQLite 记录 + Manifest 条目 + 上传文件 + BM25 索引，全部清理

### 检索策略

- **MMR 检索** — Max Marginal Relevance，兼顾相关性与多样性，避免返回重复片段（lambda_mult=0.7）
- **混合检索** — BM25 关键词匹配（jieba 分词）+ 向量语义检索，RRF（Reciprocal Rank Fusion）融合排序
- **Rerank 精排** — bge-reranker-v2-m3 CrossEncoder 对候选文档二次排序，从 top-20 精选 top-5
- **查询改写** — LLM 将口语化问题改写为更适合向量检索的精确查询（如 "LC是啥" -> "LangChain是什么框架"）

### 问答与追溯

- **智能问答** — 基于检索上下文，GLM 生成准确回答，Prompt 要求引用来源
- **来源追溯** — 回答附带：来源文件名、PDF 页码、片段索引、内容预览（前 150 字）
- **多轮对话** — SQLite 持久化对话历史，服务重启不丢失
- **对话管理** — 新建、切换、删除对话

### 工程能力

- **前后端分离** — FastAPI 后端 + Streamlit 前端，通过 HTTP 通信
- **REST API** — 7 个端点，Swagger 自动文档（/docs）
- **4 表数据库** — Conversation、Message、Document、Chunk，支持精确删除和统计
- **增量同步** — Manifest JSON 记录每个文件的 MD5、chunk 数量、更新时间
- **错误处理** — 文件格式校验、大小限制（50MB）、空文件检测、API 异常捕获
- **日志系统** — loguru 彩色控制台 + 文件轮转日志
- **一键启动** — `python run.py` 同时启动 FastAPI 和 Streamlit

## 项目结构

```
fsxm3/
├── config/
│   └── settings.py           # pydantic-settings 配置中心（从 .env 加载）
├── core/
│   ├── llm.py                # ChatOpenAI（智谱 OpenAI 兼容接口）
│   ├── embeddings.py         # bge-small-zh-v1.5（本地 Embedding）
│   ├── reranker.py           # bge-reranker-v2-m3（CrossEncoder 精排）
│   ├── vectorstore.py        # Chroma 向量库管理（增删查）
│   ├── document_processor.py # 文档加载 + 中文分块 + 元数据
│   ├── retriever.py          # 检索入口（MMR / Hybrid + Rerank）
│   ├── hybrid_retriever.py   # BM25 + Vector 混合检索（RRF 融合）
│   └── query_rewrite.py      # LLM 查询改写
├── db/
│   ├── database.py           # SQLite 引擎 + 会话管理
│   └── models.py             # ORM: Conversation / Message / Document / Chunk
├── manifest/
│   └── manager.py            # MD5 增量同步（manifest.json 读写）
├── api/
│   ├── main.py               # FastAPI 入口（lifespan 启动初始化）
│   ├── schemas.py            # Pydantic 请求/响应模型
│   ├── dependencies.py       # 依赖注入
│   └── routers/
│       ├── chat.py           # 对话 API（POST /api/chat 等）
│       └── documents.py      # 文档管理 API（上传 / 列表 / 删除）
├── ui/
│   └── app.py                # Streamlit 前端（Material Icons）
├── prompts/
│   └── templates.py          # RAG Prompt 模板（系统提示 + 用户模板）
├── data/                     # 运行时数据（自动创建，.gitignore）
│   ├── uploads/              # 上传的文档原始文件
│   ├── chroma_db/            # Chroma 持久化存储（向量 + 元数据）
│   ├── manifest.json         # 增量同步清单
│   └── knowledge_base.db     # SQLite 数据库
├── logs/                     # 日志文件（自动创建）
├── .env                      # 环境变量（API Key 等，不提交）
├── .env.example              # 环境变量模板
├── .gitignore
├── requirements.txt          # Python 依赖
├── run.py                    # 一键启动脚本
└── README.md
```

## API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/chat` | 发送消息，获取 AI 回答（支持查询改写、混合检索） |
| GET | `/api/conversations` | 列出所有对话 |
| GET | `/api/history/{id}` | 获取对话历史消息 |
| DELETE | `/api/conversations/{id}` | 删除对话及其消息 |
| POST | `/api/documents/upload` | 上传文档（含 Manifest 检查、分块、向量化） |
| GET | `/api/documents` | 列出所有已上传文档 |
| DELETE | `/api/documents/{id}` | 删除文档（Chroma + SQLite + Manifest + 文件） |

## 数据库设计（4 表）

```
Conversation          Message               Document             Chunk
+----+----------+    +----+----+------+    +----+----------+   +----+------------+------+
| id | title    |    | id | ...| role |    | id | filename |   | id | document_id| index|
|    | created  |    |    |    |      |    |    | file_path|   |    | content    |      |
|    | updated  |    | conv_id FK   |    |    | md5      |   |    | metadata   |      |
+----+----------+    +----+----+------+    |    | chunks   |   +----+------------+------+
                                           +----+----------+
```

- **Chunk 表**的作用：删除文档时精确知道哪些 chunks 需要从 Chroma 清理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入智谱 AI API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```
ZHIPUAI_API_KEY=你的API密钥
ZHIPUAI_MODEL=glm-4.6v-flash
```

### 3. 启动系统

```bash
python run.py
```

### 4. 访问

- 前端界面：http://localhost:8501
- API 文档：http://localhost:8000/docs

## 配置说明

编辑 `.env` 文件可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ZHIPUAI_MODEL` | `glm-4.6v-flash` | LLM 模型（可选 glm-4.5-air / glm-5.1） |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 本地 Embedding 模型 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Rerank 模型 |
| `CHUNK_SIZE` | `500` | 文档分块大小（字符数） |
| `CHUNK_OVERLAP` | `100` | 分块重叠大小 |
| `RETRIEVAL_TOP_K` | `20` | 向量检索召回数量 |
| `RERANK_TOP_K` | `5` | Rerank 后返回数量 |



## 常见问题

**Q: 上传超时？**
A: 检查 `.env` 中的模型 ID 是否正确，可换用 `glm-4.5-air` 测试。

**Q: 无法连接后端？**
A: 确认 FastAPI 已启动：访问 http://localhost:8000/docs 看到 Swagger 页面即正常。

**Q: 回答不准确？**
A: 尝试开启"混合检索"和"查询改写"功能，或优化提问方式。

**Q: Reranker 首次加载慢？**
A: 首次使用需加载模型（约 2.2GB），后续从本地缓存加载，秒开。

**Q: 如何更换模型？**
A: 编辑 `.env` 中的 `ZHIPUAI_MODEL`，然后重启服务。
