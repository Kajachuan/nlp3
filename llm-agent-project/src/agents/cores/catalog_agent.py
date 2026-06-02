from __future__ import annotations

from dataclasses import dataclass

from src.rag.retrievers.catalog_retriever import CatalogHit, CatalogRetriever


@dataclass(frozen=True)
class CatalogAgentResult:
    answer: str
    hits: list[CatalogHit]
    self_review: str


class CatalogAgent:
    role = "catalog_rag_agent"

    def __init__(self, retriever: CatalogRetriever) -> None:
        self.retriever = retriever

    def run(self, query: str, top_k: int = 3) -> CatalogAgentResult:
        hits = self.retriever.search(query, top_k=top_k)
        if not hits:
            return CatalogAgentResult(
                answer="No encontre componentes del catalogo que coincidan con la consulta.",
                hits=[],
                self_review="Baja cobertura: conviene ampliar catalogo o reformular la consulta.",
            )

        lines = []
        for index, hit in enumerate(hits, start=1):
            item = hit.item
            lines.append(
                f"{index}. {item['name']} ({item['sku']}): {item['description']} "
                f"Uso sugerido: {item['recommended_use']} Stock: {item['stock']}. "
                f"Precio aprox.: USD {item['price_usd']}."
            )

        review = self._review(hits)
        return CatalogAgentResult(answer="\n".join(lines), hits=hits, self_review=review)

    def _review(self, hits: list[CatalogHit]) -> str:
        best_score = hits[0].score if hits else 0
        if best_score < 0.04:
            return "Confianza media-baja: los terminos recuperados son pocos."
        if hits[0].item.get("stock", 0) < 20:
            return "Advertencia: la mejor opcion tiene stock bajo."
        return "Confianza aceptable: hay coincidencia lexical y stock disponible."
