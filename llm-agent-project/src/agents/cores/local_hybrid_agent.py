from __future__ import annotations

from dataclasses import dataclass

from src.rag.retrievers.hybrid_retriever import HybridCatalogRetriever, HybridSearchResponse


@dataclass(frozen=True)
class LocalHybridAgentResult:
    response: HybridSearchResponse
    answer: str
    self_review: str


class LocalHybridSearchAgent:
    role = "local_hybrid_search_agent"

    def __init__(self, retriever: HybridCatalogRetriever) -> None:
        self.retriever = retriever

    def run(self, query: str, top_k: int = 3) -> LocalHybridAgentResult:
        response = self.retriever.search(query, top_k=top_k)
        if not response.results:
            return LocalHybridAgentResult(
                response=response,
                answer="No local catalog matches were found.",
                self_review="Local evidence is insufficient; web fallback may be required.",
            )

        lines = []
        for index, result in enumerate(response.results, start=1):
            item = result.item
            lines.append(
                f"{index}. {item['name']} ({item['sku']}): price USD {item.get('price_usd')}, "
                f"stock {item.get('stock')}, score {result.score}, sources {', '.join(result.sources)}."
            )
        return LocalHybridAgentResult(
            response=response,
            answer="\n".join(lines),
            self_review="Local evidence available." if response.results else "No local evidence.",
        )
