from __future__ import annotations

from dataclasses import dataclass

from src.tools.external.web_search_adapter import WebResult, WebSearchAdapter, WebSearchMethod


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

    def run(
        self,
        query: str,
        limit: int = 3,
        method: WebSearchMethod | None = None,
        preferred_sites: list[str] | None = None,
        preferred_only: bool = False,
        include_delivery_details: bool = False,
    ) -> WebResearchResult:
        results = self.adapter.search_with_options(
            query=query,
            limit=limit,
            method=method,
            preferred_sites=preferred_sites,
            preferred_only=preferred_only,
            include_delivery_details=include_delivery_details,
        )
        if not results:
            return WebResearchResult(
                answer="No se obtuvieron resultados externos.",
                results=[],
                self_review="Sin evidencia externa disponible.",
                tool_mode=self.adapter.last_mode,
            )

        answer = "\n".join(
            f"{idx}. {result.title}{self._commercial_details(result)}: {result.snippet} ({result.url})"
            for idx, result in enumerate(results, start=1)
        )
        review = "Evidencia externa disponible; verificar enlaces antes de usar en produccion."
        if self.adapter.last_mode.startswith("offline-demo"):
            review = "Modo demo sin red: sirve para validar el flujo, no como fuente real."
        return WebResearchResult(answer=answer, results=results, self_review=review, tool_mode=self.adapter.last_mode)

    def _commercial_details(self, result: WebResult) -> str:
        details = []
        if result.vendor:
            details.append(f"vendor={result.vendor}")
        if result.price:
            details.append(f"precio={result.price}")
        if result.score is not None:
            details.append(f"score={result.score:.2f}")
        return f" ({', '.join(details)})" if details else ""
