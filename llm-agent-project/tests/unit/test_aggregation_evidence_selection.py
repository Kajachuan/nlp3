from pathlib import Path

from src.agents.cores.local_hybrid_agent import LocalHybridAgentResult
from src.agents.cores.web_research_agent import WebResearchResult
from src.agents.orchestration.component_advisor import ComponentAdvisorOrchestrator, PlannerDecision
from src.agents.orchestration.evidence_gate import LocalEvidenceGate
from src.rag.retrievers.hybrid_retriever import HybridSearchResponse, HybridSearchResult
from src.tools.external.web_search_adapter import WebResult


def _orchestrator() -> ComponentAdvisorOrchestrator:
    return ComponentAdvisorOrchestrator(
        local_hybrid_agent=None,  # type: ignore[arg-type]
        web_agent=None,  # type: ignore[arg-type]
        evidence_gate=LocalEvidenceGate(),
        audit_log_path=Path("unused.jsonl"),
    )


def test_fallback_aggregator_prefers_web_when_local_misses_specific_part_number() -> None:
    local = LocalHybridAgentResult(
        response=HybridSearchResponse(
            results=[
                HybridSearchResult(
                    item={
                        "sku": "MCU-ESP8266-RP2040-PICO-19",
                        "name": "Placa MCU RP2040 Pico",
                        "category": "microcontroller",
                        "description": "Placa de desarrollo MCU.",
                        "stock": 42,
                        "price_usd": 7.4,
                    },
                    score=0.3183,
                    semantic_score=0.3183,
                    keyword_score=0.0,
                    matched_terms=[],
                    sources=["rag"],
                    document="Placa MCU RP2040 Pico microcontroller",
                )
            ],
            rag_hits=[],
            keyword_hits=[],
            timings_ms={},
        ),
        answer="bad local answer",
        self_review="bad local evidence",
    )
    web = WebResearchResult(
        answer="web answer",
        results=[
            WebResult(
                title="Allegro MicroSystems ACS71240 current sensor",
                url="https://www.google.com/search?ibp=oshop&q=acs71240",
                snippet="ACS71240 current sensor",
                source="web",
                price="2.83",
                vendor="web",
            )
        ],
        self_review="web evidence",
        tool_mode="web_search_tool",
    )
    planner = PlannerDecision(
        route="local_search",
        normalized_query="Precio de acs71240 de allegro",
        product_terms=["acs71240", "allegro"],
        web_search_query="acs71240 allegro",
        needs_price=True,
        stream_message="",
        reason="test",
        model="test",
        provider="test",
    )

    decision = _orchestrator()._fallback_aggregator(local, web, planner, "test", "test")

    assert decision.product_name == "Allegro MicroSystems ACS71240 current sensor"
    assert decision.purchase_url == "https://www.google.com/search?ibp=oshop&q=acs71240"
    assert "RP2040" not in decision.answer
    assert "2.83" in decision.answer
    assert "https://www.google.com/search?ibp=oshop&q=acs71240" in decision.answer


def test_local_supports_specific_term_when_part_number_is_present() -> None:
    local = LocalHybridAgentResult(
        response=HybridSearchResponse(
            results=[
                HybridSearchResult(
                    item={
                        "sku": "SENSOR-ACS71240",
                        "name": "ACS71240 current sensor",
                        "category": "sensor",
                        "description": "Hall current sensor",
                    },
                    score=0.5,
                    semantic_score=0.5,
                    keyword_score=0.0,
                    matched_terms=["acs71240"],
                    sources=["rag"],
                    document="ACS71240 Hall current sensor",
                )
            ],
            rag_hits=[],
            keyword_hits=[],
            timings_ms={},
        ),
        answer="local",
        self_review="local",
    )
    planner = PlannerDecision(
        route="local_search",
        normalized_query="Precio de acs71240",
        product_terms=["acs71240"],
        web_search_query="acs71240",
        needs_price=True,
        stream_message="",
        reason="test",
        model="test",
        provider="test",
    )

    assert _orchestrator()._local_supports_specific_terms(local, planner)
