from __future__ import annotations

from dataclasses import dataclass

from src.tools.external.web_search_adapter import WebResult, WebSearchAdapter


@dataclass(frozen=True)
class WebResearchResult:
    answer: str
    results: list[WebResult]
    self_review: str
    tool_mode: str


class WebResearchAgent:
    role = "web_research_agent"

    def __init__(self, adapter: WebSearchAdapter) -> None:
        self.adapter = adapter

    def run(self, query: str, limit: int = 3) -> WebResearchResult:
        results = self.adapter.search(query, limit=limit)
        if not results:
            return WebResearchResult(
                answer="No se obtuvieron resultados externos.",
                results=[],
                self_review="Sin evidencia externa disponible.",
                tool_mode=self.adapter.mode,
            )

        answer = "\n".join(f"{idx}. {result.title}: {result.snippet} ({result.url})" for idx, result in enumerate(results, start=1))
        review = "Evidencia externa disponible; verificar enlaces antes de usar en produccion."
        if self.adapter.mode == "offline-demo":
            review = "Modo demo sin red: sirve para validar el flujo, no como fuente real."
        return WebResearchResult(answer=answer, results=results, self_review=review, tool_mode=self.adapter.mode)
