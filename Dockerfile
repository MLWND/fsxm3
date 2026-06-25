FROM python:3.11-slim

WORKDIR /app

# 阿里云镜像（国内加速 apt 下载）
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.aliyun.com/debian/ trixie main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ trixie-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security trixie-security main contrib non-free" >> /etc/apt/sources.list

# 系统依赖（只装 curl，Python 包全部走预编译 wheel，无需 build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1) 用 curl 下载 CPU-only torch（pip 对 PyTorch CDN 超时，curl 正常）
COPY requirements.txt .
RUN curl -sL -o /tmp/torch-2.12.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl \
    "https://download.pytorch.org/whl/cpu/torch-2.12.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" && \
    pip install --no-cache-dir /tmp/torch-2.12.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
    rm /tmp/torch-2.12.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl && \
    python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2) 再装其余依赖（torch 已存在，pip 不会再拉）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    $(grep -v '^torch' requirements.txt | grep -v '^--extra-index' | grep -v '^#' | grep -v '^$' | tr '\n' ' ')

# HuggingFace 镜像（国内加速）
ENV HF_ENDPOINT=https://hf-mirror.com

# 预下载 Embedding 模型（hf-mirror.com 308 重定向导致 huggingface_hub 失败，用 curl -L 手动下载）
RUN MODEL_DIR="/app/models/bge-small-zh-v1.5" && \
    mkdir -p "$MODEL_DIR" && \
    BASE="https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main" && \
    for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt model.safetensors; do \
        echo "Downloading $f ..." && \
        curl -sL -o "$MODEL_DIR/$f" "$BASE/$f" || exit 1; \
    done && \
    echo "Model downloaded successfully" && \
    ls -lh "$MODEL_DIR/"

# 预下载 jieba 词典
RUN python -c "import jieba; jieba.initialize()"

# 恢复 HuggingFace 镜像（运行时也用镜像，防止离线模式下仍尝试连接官方源）
ENV HF_ENDPOINT=https://hf-mirror.com

# 应用代码
COPY . .

# 入口脚本权限
RUN chmod +x docker-entrypoint.sh

# 创建数据目录
RUN mkdir -p data/uploads data/chroma_db data/logs logs

# 离线模式
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
