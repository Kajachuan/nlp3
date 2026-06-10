from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL_OPTIONS = [
    "canopylabs/orpheus-arabic-saudi",
    "canopylabs/orpheus-v1-english",
    "groq/compound",
    "groq/compound-mini",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-prompt-guard-2-22m",
    "meta-llama/llama-prompt-guard-2-86m",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
]


def parse_model_options(value: str | None = None) -> list[str]:
    raw = value if value is not None else os.getenv("LLM_MODEL_OPTIONS", "")
    options = [item.strip() for item in raw.split(",") if item.strip()]
    return options or DEFAULT_MODEL_OPTIONS


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMModelSelection:
    planner_model: str
    direct_response_model: str
    verifier_model: str


def default_model_selection() -> LLMModelSelection:
    options = parse_model_options()
    planner = os.getenv("PLANNER_MODEL", "openai/gpt-oss-120b")
    direct = os.getenv("DIRECT_RESPONSE_MODEL", planner)
    verifier = os.getenv("VERIFIER_MODEL", "groq/compound")
    valid = set(options)
    return LLMModelSelection(
        planner_model=planner if planner in valid else options[0],
        direct_response_model=direct if direct in valid else options[0],
        verifier_model=verifier if verifier in valid else options[0],
    )
