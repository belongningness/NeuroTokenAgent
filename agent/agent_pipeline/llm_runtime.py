"""LLM configuration and model construction for the Agent Server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "openai:gpt-latest"
DEFAULT_CONFIG = "agent_pipeline/configs/camcan_meg_h5.toml"
DEFAULT_LLM_ENV_FILE = str(Path(__file__).resolve().parent / "configs" / "llm_env.json")


def load_llm_env_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "llm" in data:
        return data["llm"]
    return data


def configure_llm_env(api_key: str | None, base_url: str | None, proxy: str | None) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy


def make_llm(model: str):
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, temperature=0)
