# 智能知识库问答系统

基于 LangChain + RAG 技术的智能知识库问答系统。支持文档上传、自动索引、混合检索、Rerank 精排、智能问答。

## 技术栈

| 模块 | 选择 |
|------|------|
| LLM | GLM-4.6V-Flash (智谱 AI) |
| RAG 框架 | LangChain |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 检索策略 | MMR / BM25+向量混合检索 |
| 向量库 | Chroma |
| 前端 | Streamlit |
| 后端 | FastAPI |
| 数据库 | SQLite |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env`，填入智谱 AI API Key：

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

## 功能特性

- **多格式支持** — TXT / PDF / DOCX 文档上传
- **MMR 检索** — 兼顾相关性与多样性，避免重复片段
- **混合检索** — BM25 关键词 + 向量语义，RRF 融合排序
- **Rerank 精排** — CrossEncoder 二次排序，提升检索准确率
- **查询改写** — LLM 将口语化问题改写为精确检索查询
- **来源追溯** — 回答附带来源文件、页码和内容片段
- **智能问答** — 基于文档内容生成回答
- **多轮对话** — 支持上下文连续提问，历史持久化
- **增量同步** — MD5 检测，避免重复 Embedding
- **精确删除** — 删除文档时精准清理向量和元数据

## 项目结构

```
fsxm3/
├── config/
│   └── settings.py           # pydantic-settings 配置中心
├── core/
│   ├── llm.py                # ChatOpenAI (智谱 OpenAI 兼容接口)
│   ├── embeddings.py         # bge-small-zh-v1.5 (本地)
│   ├── reranker.py           # bge-reranker-v2-m3 (CrossEncoder)
│   ├── vectorstore.py        # Chroma 向量库管理
│   ├── document_processor.py # 文档加载 + 中文分块
│   ├── retriever.py          # MMR / Hybrid + Rerank pipeline
│   ├── hybrid_retriever.py   # BM25 + Vector 混合检索
│   └── query_rewrite.py      # LLM 查询改写
├── db/
│   ├── database.py           # SQLite 引擎
│   └── models.py             # ORM: Conversation/Message/Document/Chunk
├── manifest/
│   └── manager.py            # MD5 增量同步
├── api/
│   ├── main.py               # FastAPI 入口
│   ├── schemas.py            # Pydantic 模型
│   └── routers/
│       ├── chat.py           # 对话 API
│       └── documents.py      # 文档管理 API
├── ui/
│   └── app.py                # Streamlit 前端
├── prompts/
│   └── templates.py          # RAG Prompt 模板
├── data/                     # 运行时数据（自动创建）
├── run.py                    # 一键启动脚本
└── requirements.txt
```

## 架构流程

```
用户上传文档
  -> 格式校验 + MD5 Manifest 检查
  -> 文本提取 + 中文分块
  -> Embedding 向量化 (bge-small-zh-v1.5)
  -> 存入 Chroma + SQLite + Manifest

用户提问
  -> [可选] 查询改写 (LLM Rewrite)
  -> 向量检索 top-20 (MMR 或 BM25+Vector 混合)
  -> Rerank 精排 (bge-reranker-v2-m3) -> top-5
  -> 组装上下文 + Prompt
  -> GLM-4.6V-Flash 生成回答
  -> 返回答案 + 来源文件 + 页码 + 片段预览
```

## 版本演进

| 版本 | 功能 | 状态 |
|------|------|------|
| V1 | Embedding + Chroma + MMR + GLM 问答 | 完成 |
| V2 | + 来源追溯 + Query Rewrite 查询改写 | 完成 |
| V3 | + BM25 混合检索 + Rerank 精排 | 完成 |
| V4 | + LangGraph Agent (文档评分 + 多步推理) | 计划 |

## 更换模型

编辑 `.env` 文件中的 `ZHIPUAI_MODEL`：

```
ZHIPUAI_MODEL=glm-4.5-air    # 快速调试
ZHIPUAI_MODEL=glm-5.1        # 正式使用
```

然后重启服务。

## 常见问题

**Q: 上传超时？**
A: 检查 `.env` 中的模型 ID 是否正确，可换用 `glm-4.5-air` 测试。

**Q: 无法连接后端？**
A: 确认 FastAPI 已启动：访问 http://localhost:8000/docs 看到 Swagger 页面即正常。

**Q: 回答不准确？**
A: 尝试开启"混合检索"和"查询改写"功能，或优化提问方式。

**Q: Reranker 首次加载慢？**
A: 首次使用需加载模型（约 2.2GB），后续从本地缓存加载，秒开。
