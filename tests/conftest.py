"""共享测试 fixtures。"""

import os
import tempfile
from pathlib import Path

import pytest

# 确保测试环境不读写真实 .env
os.environ.setdefault("ZHIPUAI_API_KEY", "test-key-not-real")


@pytest.fixture
def tmp_txt(tmp_path):
    """创建临时 TXT 文件。"""
    p = tmp_path / "test.txt"
    p.write_text("这是一段测试文本。\n第二行内容。", encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_gbk_txt(tmp_path):
    """创建 GBK 编码的 TXT 文件。"""
    p = tmp_path / "test_gbk.txt"
    p.write_bytes("这是GBK编码的文本。\n第二行。".encode("gbk"))
    return str(p)


@pytest.fixture
def tmp_csv(tmp_path):
    """创建临时 CSV 文件。"""
    p = tmp_path / "test.csv"
    p.write_text("姓名,年龄,城市\n张三,25,北京\n李四,30,上海\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def tmp_markdown(tmp_path):
    """创建临时 Markdown 文件。"""
    p = tmp_path / "test.md"
    p.write_text("# 标题\n\n这是**加粗**内容。\n\n## 二级标题\n\n列表项1\n", encoding="utf-8")
    return str(p)
