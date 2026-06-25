#!/bin/bash
set -e

# 离线模式：模型已预下载，不联网检查
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "========================================="
echo "  Knowledge Base Q&A System - Starting"
echo "========================================="

# 启动 FastAPI 后台
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "[1/2] FastAPI started -> http://localhost:8000/docs"

# 等待 FastAPI 就绪
sleep 5

# 启动 Streamlit 前台（保持容器运行）
python -m streamlit run ui/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
UI_PID=$!
echo "[2/2] Streamlit started -> http://localhost:8501"
echo ""
echo "All services running. Ctrl+C to stop."

# 信号处理：优雅关闭
cleanup() {
    echo "Stopping services..."
    kill $API_PID $UI_PID 2>/dev/null
    wait $API_PID $UI_PID 2>/dev/null
    echo "All services stopped."
}
trap cleanup SIGTERM SIGINT

# 等待任一进程退出
wait -n $API_PID $UI_PID
cleanup
