from __future__ import annotations

import os
import re
import time
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


def _require_langfuse() -> None:
    _load_real_env()
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        pytest.skip("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for Langfuse e2e tests.")


def _real_models() -> LLMModelSelection:
    model = os.getenv("E2E_GROQ_MODEL") or os.getenv("PLANNER_MODEL") or "llama-3.3-70b-versatile"
    return LLMModelSelection(
        planner_model=model,
        direct_response_model=os.getenv("DIRECT_RESPONSE_MODEL") or model,
        verifier_model=os.getenv("VERIFIER_MODEL") or model,
    )


def _fetch_langfuse_trace(trace_id: str):
    from langfuse import Langfuse

    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    client = Langfuse(host=host)
    for _ in range(8):
        try:
            trace = client.fetch_trace(trace_id)
        except Exception:
            trace = None
        if trace:
            return trace
        time.sleep(5)
    return None


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


@pytest.mark.e2e
def test_real_api_multi_product_uses_local_rag_and_web_fallback_with_two_prices() -> None:
    _require_groq()
    _require_serpapi()
    _require_langfuse()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    response = advisor.answer(
        "Dame el precio del sensor BME280 y del sensor de corriente ACS71240 de Allegro",
        top_k=1,
        web_limit=3,
        web_method="google_shopping",
        model_selection=_real_models(),
        max_products=3,
    )

    assert response.ok
    assert response.planner_decision is not None
    assert response.planner_decision.provider == "groq"
    assert response.planner_decision.route == "local_search"
    assert response.multi_product
    assert response.processed_product_count == 2
    assert len(response.product_evidence) == 2

    bme_bundle = next(
        bundle for bundle in response.product_evidence if "bme280" in bundle.product.normalized_query.lower()
    )
    acs_bundle = next(
        bundle for bundle in response.product_evidence if "acs71240" in bundle.product.normalized_query.lower()
    )

    assert bme_bundle.local.response.results
    assert bme_bundle.evidence_gate.use_web_fallback is False
    assert bme_bundle.local.response.results[0].item.get("price_usd") not in {None, ""}

    assert acs_bundle.evidence_gate.use_web_fallback is True
    assert acs_bundle.web is not None
    assert acs_bundle.web.tool_mode == "web_search_tool"
    assert any(result.price for result in acs_bundle.web.results)

    answer = response.final_answer.lower()
    assert "bme280" in answer
    assert "acs71240" in answer
    assert str(bme_bundle.local.response.results[0].item["price_usd"]) in response.final_answer
    assert len(re.findall(r"(?i)\bprecio\s*:", response.final_answer)) >= 2

    trace_id = response.metrics["labels"].get("langfuse_trace_id")
    assert trace_id
    trace = _fetch_langfuse_trace(str(trace_id))
    assert trace is not None
