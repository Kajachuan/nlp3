from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.runtime.config import LLMModelSelection
from src.runtime.pipelines.factory import build_component_advisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_real_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _require_groq() -> None:
    _load_real_env()
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required for real API e2e tests.")


def _require_serpapi() -> None:
    _load_real_env()
    if not os.getenv("SERPAPI_API_KEY"):
        pytest.skip("SERPAPI_API_KEY is required for real web fallback e2e tests.")


def _real_models() -> LLMModelSelection:
    model = os.getenv("E2E_GROQ_MODEL") or os.getenv("PLANNER_MODEL") or "llama-3.3-70b-versatile"
    return LLMModelSelection(
        planner_model=model,
        direct_response_model=os.getenv("DIRECT_RESPONSE_MODEL") or model,
        verifier_model=os.getenv("VERIFIER_MODEL") or model,
    )


@pytest.mark.e2e
def test_real_api_local_product_flow_uses_groq_and_local_catalog() -> None:
    _require_groq()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    response = advisor.answer(
        "Dame precio y disponibilidad del sensor BME280",
        top_k=3,
        web_limit=1,
        model_selection=_real_models(),
    )

    assert response.ok
    assert response.planner_decision is not None
    assert response.planner_decision.provider == "groq"
    assert response.planner_decision.route == "local_search"
    assert response.aggregator_decision is not None
    assert response.aggregator_decision.provider == "groq"
    assert response.local is not None
    assert response.metrics["counters"]["rag_result_count"] > 0
    assert response.metrics["counters"]["keyword_result_count"] > 0
    assert response.metrics["labels"]["web_fallback_used"] is False
    assert "BME280" in response.final_answer


@pytest.mark.e2e
def test_real_api_agent_info_flow_uses_groq_without_tools() -> None:
    _require_groq()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    response = advisor.answer(
        "Como funciona este agente y para que sirve?",
        top_k=3,
        web_limit=1,
        model_selection=_real_models(),
    )

    assert response.ok
    assert response.planner_decision is not None
    assert response.planner_decision.provider == "groq"
    assert response.planner_decision.route == "agent_info"
    assert response.local is None
    assert response.web is None
    assert response.final_answer.strip()
    assert "rag_result_count" not in response.metrics["counters"]
    assert response.metrics["labels"]["web_fallback_used"] is False


@pytest.mark.e2e
def test_real_api_out_of_domain_flow_uses_groq_without_tools() -> None:
    _require_groq()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    response = advisor.answer(
        "Quien gano el mundial de futbol 2014?",
        top_k=3,
        web_limit=1,
        model_selection=_real_models(),
    )

    assert response.ok
    assert response.planner_decision is not None
    assert response.planner_decision.provider == "groq"
    assert response.planner_decision.route == "out_of_domain"
    assert response.local is None
    assert response.web is None
    assert response.final_answer.strip()
    assert "rag_result_count" not in response.metrics["counters"]
    assert response.metrics["labels"]["web_fallback_used"] is False


@pytest.mark.e2e
def test_real_api_web_fallback_uses_serpapi_when_local_catalog_is_insufficient() -> None:
    _require_groq()
    _require_serpapi()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    response = advisor.answer(
        "Dame el precio del LM741",
        top_k=1,
        web_limit=3,
        web_method="google_shopping",
        model_selection=_real_models(),
    )

    assert response.ok
    assert response.planner_decision is not None
    assert response.planner_decision.provider == "groq"
    assert response.planner_decision.web_search_query
    assert response.planner_decision.web_search_query.lower() != "dame el precio del lm741"
    assert response.metrics["labels"]["web_fallback_used"] is True
    assert response.web is not None
    assert response.web.tool_mode == "web_search_tool"
    assert response.metrics["counters"]["web_result_count"] > 0
