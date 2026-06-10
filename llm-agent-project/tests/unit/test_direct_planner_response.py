from __future__ import annotations

import json
from pathlib import Path
from src.agents.orchestration.component_advisor import ComponentAdvisorOrchestrator
from src.runtime.providers.groq_llm import LLMResult


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, model: str, system_prompt: str, user_payload: str) -> LLMResult:
        self.calls += 1
        return LLMResult(
            content=json.dumps(
                {
                    "route": "out_of_domain",
                    "normalized_query": "Cuanto gana Messi",
                    "product_terms": [],
                    "web_search_query": "messi earnings",
                    "needs_price": False,
                    "stream_message": "Out of scope.",
                    "reason": "Sports/celebrity earnings are outside electronics purchasing.",
                    "direct_answer": "This agent is only for finding electronic components to buy.",
                }
            ),
            model=model,
            provider="fake",
            prompt_tokens=10,
            completion_tokens=5,
            estimated_cost=0.00001,
        )


def test_out_of_domain_uses_planner_direct_answer_without_second_llm_call() -> None:
    provider = FakeLLMProvider()
    audit_log = Path("tmp") / "test_direct_planner_response_audit.jsonl"
    advisor = ComponentAdvisorOrchestrator(
        local_hybrid_agent=None,  # type: ignore[arg-type]
        web_agent=None,  # type: ignore[arg-type]
        evidence_gate=None,  # type: ignore[arg-type]
        audit_log_path=audit_log,
        llm_provider=provider,  # type: ignore[arg-type]
    )

    response = advisor.answer("Cuanto gana Messi?")

    assert provider.calls == 1
    assert response.planner_decision is not None
    assert response.planner_decision.route == "out_of_domain"
    assert response.final_answer == "This agent is only for finding electronic components to buy."
    assert "direct_response" not in response.metrics["timings_ms"]
    assert "direct_response_total_tokens" not in response.metrics["counters"]
    assert response.local is None
    assert response.web is None
