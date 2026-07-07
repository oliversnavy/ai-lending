from __future__ import annotations
import pathlib
import yaml
from langchain_openai import ChatOpenAI

from ..models.episode import TreatmentConfig

_CONFIG: dict | None = None


def _cfg() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = pathlib.Path("configs/base.yaml")
        with open(path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_primary_client(treatment_config: TreatmentConfig) -> ChatOpenAI:
    m = _cfg()["models"]
    if treatment_config.primary_is_27b:
        url, model, max_tokens = m["advisor_url"], m["advisor_model"], m["advisor_max_tokens"]
    else:
        url, model, max_tokens = m["primary_url"], m["primary_model"], m["primary_max_tokens"]

    return ChatOpenAI(
        base_url=url,
        api_key="not-needed",
        model=model,
        max_tokens=max_tokens,
        temperature=0.6,
    )


def get_advisor_client() -> ChatOpenAI:
    m = _cfg()["models"]
    return ChatOpenAI(
        base_url=m["advisor_url"],
        api_key="not-needed",
        model=m["advisor_model"],
        max_tokens=m["advisor_max_tokens"],
        temperature=0.7,
    )
