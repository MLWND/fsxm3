"""测试配置校验。"""

import warnings

import pytest


def test_settings_loads_with_env():
    """配置能正常加载。"""
    from config.settings import settings
    assert settings.ZHIPUAI_MODEL  # 非空
    assert settings.ZHIPUAI_BASE_URL.startswith("http")


def test_settings_warns_on_empty_api_key():
    """空 API Key 应产生警告。"""
    from config.settings import Settings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = Settings(ZHIPUAI_API_KEY="")
        # model_validator 会触发警告
        assert any("ZHIPUAI_API_KEY" in str(warning.message) for warning in w)


def test_settings_cors_origins_default():
    """CORS 默认值应包含 localhost:8501。"""
    from config.settings import settings
    assert "http://localhost:8501" in settings.CORS_ORIGINS


def test_settings_history_max_tokens():
    """历史 token 上限应为正整数。"""
    from config.settings import settings
    assert settings.HISTORY_MAX_TOKENS > 0
