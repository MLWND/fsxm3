"""一键启动脚本：同时启动 FastAPI 和 Streamlit。"""

import subprocess
import sys
import time


def main():
    print("=" * 50)
    print("  智能知识库问答系统 - 启动中...")
    print("=" * 50)

    # 启动 FastAPI
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=".",
    )
    print("[1/2] FastAPI 后端已启动 → http://localhost:8000/docs")

    time.sleep(3)

    # 启动 Streamlit
    ui_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", "8501"],
        cwd=".",
    )
    print("[2/2] Streamlit 前端已启动 → http://localhost:8501")
    print()
    print("按 Ctrl+C 停止所有服务")

    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        api_proc.terminate()
        ui_proc.terminate()
        api_proc.wait()
        ui_proc.wait()
        print("已停止 ✓")


if __name__ == "__main__":
    main()
