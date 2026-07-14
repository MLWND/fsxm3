# Docker 部署指南

## 前置条件

1. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows / Mac / Linux 均可）
2. 确认 Docker 已启动：终端输入 `docker --version` 能正常输出

## 快速开始

### 第 1 步：配置 API Key

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`，将 `sk-你的API-Key` 替换为你的 Agnes AI API Key，并（可选）填入 Exa API Key 以启用联网搜索：

```
ZHIPUAI_API_KEY=sk-xxxxxxxxxxxxx
ZHIPUAI_MODEL=agnes-2.0-flash
ZHIPUAI_BASE_URL=https://apihub.agnes-ai.com/v1
EXA_API_KEY=your-exa-api-key      # 可选，启用联网搜索
```

> API Key 获取：Agnes AI → https://apihub.agnes-ai.com 控制台；Exa → https://dashboard.exa.ai/api-keys（免费额度）

### 第 2 步：构建并启动

```bash
docker compose up -d --build
```

首次构建约 15-25 分钟（下载 Python 依赖 + Embedding 模型），后续启动仅需 30 秒。

> **国内网络提示**：Dockerfile 已配置阿里云 Debian 镜像、清华 PyPI 镜像和 hf-mirror.com 模型镜像。如果构建仍然很慢，参考下方[网络问题](#网络问题)章节。

### 第 3 步：访问

| 服务 | 地址 |
|------|------|
| **问答界面** | http://localhost:8501 |
| **API 文档** | http://localhost:8000/docs |
| **健康检查** | http://localhost:8000/health |

## 使用方式

1. 打开 http://localhost:8501
2. 左侧上传文档（支持 TXT / PDF / DOCX / CSV / Markdown / Excel）
3. 底部输入框提问，系统基于文档内容回答
4. 可开启「流式输出」逐字显示、开启「混合检索」提升质量
5. 配置 `EXA_API_KEY` 后，可在侧边栏切换搜索模式（仅本地 / 仅网络 / 两者混合）

## 常用命令

```bash
# 查看运行状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 重新构建（代码更新后）
docker compose up -d --build

# 重新构建（不使用缓存，完全重装依赖）
docker compose build --no-cache && docker compose up -d

# 删除所有数据（含上传文件和向量库）
docker compose down -v
```

## GPU 加速（可选）

如果你的电脑有 NVIDIA 显卡，可以启用 GPU 加速 Embedding 模型：

1. 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. 编辑 `docker-compose.yml`，取消 GPU 部分的注释：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. 重启服务：`docker compose up -d`

> 注意：Docker 内的 torch 为 CPU 版本。GPU 加速仅对 Embedding 模型生效。本地运行（非 Docker）自动使用 CUDA 版 torch。

## 数据说明

- 上传的文件、向量数据库、SQLite 数据库都持久化在 Docker volume `app-data` 中
- 容器重启不丢失数据
- 彻底清除数据：`docker compose down -v`

## 网络问题

### Docker 镜像加速

如果 Docker 基础镜像拉取很慢，配置 Docker 镜像源（Docker Desktop → Settings → Docker Engine）：

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.m.daocloud.io",
    "https://docker.xuanyuan.me"
  ]
}
```

修改后需要 **退出并重启 Docker Desktop** 才能生效。

### pip 安装超时

Dockerfile 默认使用清华 PyPI 镜像（`pypi.tuna.tsinghua.edu.cn`）。如果仍然很慢，可以改为其他镜像：

| 镜像 | 地址 |
|------|------|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |

### Embedding 模型下载失败

模型文件在构建时通过 curl 从 hf-mirror.com 下载到镜像内。如果下载失败：

```bash
# 完全不使用缓存重新构建
docker compose build --no-cache && docker compose up -d
```

### 端口被占用

修改 `docker-compose.yml` 中的端口映射：

```yaml
    ports:
      - "9000:8000"   # 改为 9000
      - "9001:8501"   # 改为 9001
```

## 构建细节

Dockerfile 的构建流程：

1. **基础镜像** — `python:3.11-slim`（Debian 13）
2. **系统依赖** — 只安装 curl（用于健康检查），无需 build-essential
3. **CPU torch** — 从 PyTorch CDN 下载 CPU-only torch（curl 绕过 pip 超时）
4. **Python 依赖** — 从清华 PyPI 安装其余包
5. **Embedding 模型** — 从 hf-mirror.com 下载 bge-small-zh-v1.5（curl -L 跟随重定向）
6. **jieba 词典** — 预初始化
7. **应用代码** — COPY 进镜像

最终镜像约 3GB（主要是 torch + transformers + embedding 模型）。
