from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0


class GroqLLMProvider:
    def __init__(self) -> None:
        self.available = bool(os.getenv("GROQ_API_KEY"))

    def invoke(self, model: str, system_prompt: str, user_payload: str) -> LLMResult:
        if not self.available:
            return LLMResult(content="", model=model, provider="rules-fallback")
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatGroq(model=model, temperature=0)
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_payload)])
            content = str(getattr(response, "content", ""))
            usage = getattr(response, "usage_metadata", {}) or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            estimated_cost = round((input_tokens / 1_000_000) * 1.0 + (output_tokens / 1_000_000) * 5.0, 6)
            return LLMResult(
                content=content,
                model=model,
                provider="groq",
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                estimated_cost=estimated_cost,
            )
        except Exception as exc:
            return LLMResult(content=json.dumps({"error": str(exc)}), model=model, provider="rules-fallback")


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
