from __future__ import annotations

from dataclasses import dataclass
import re

from src.rag.retrievers.hybrid_retriever import HybridSearchResponse
from src.runtime.config import env_bool, env_float, env_int


@dataclass(frozen=True)
class EvidenceGateDecision:
    use_web_fallback: bool
    reason: str
    best_local_score: float


class LocalEvidenceGate:
    def __init__(
        self,
        min_score: float | None = None,
        min_results: int | None = None,
        web_fallback_enabled: bool | None = None,
    ) -> None:
        self.min_score = min_score if min_score is not None else env_float("LOCAL_SEARCH_MIN_SCORE", 0.08)
        self.min_results = min_results if min_results is not None else env_int("LOCAL_SEARCH_MIN_RESULTS", 1)
        self.web_fallback_enabled = (
            web_fallback_enabled if web_fallback_enabled is not None else env_bool("WEB_FALLBACK_ENABLED", True)
        )

    def decide(
        self,
        local: HybridSearchResponse,
        needs_price: bool = True,
        needs_shipping: bool = True,
        required_terms: list[str] | None = None,
    ) -> EvidenceGateDecision:
        best_score = local.results[0].score if local.results else 0.0
        if not self.web_fallback_enabled:
            return EvidenceGateDecision(False, "web fallback disabled", best_score)
        if len(local.results) < self.min_results:
            return EvidenceGateDecision(True, "not enough local results", best_score)
        if best_score < self.min_score:
            return EvidenceGateDecision(True, "best local score below threshold", best_score)

        specific_terms = self._specific_product_terms(required_terms or [])
        if specific_terms and not self._best_result_contains_any(local, specific_terms):
            return EvidenceGateDecision(True, "specific product terms missing from local best result", best_score)

        best = local.results[0].item
        if needs_price and best.get("price_usd") in {None, ""}:
            return EvidenceGateDecision(True, "local result is missing price", best_score)
        if needs_shipping:
            # The local demo catalog has stock but no shipping ETA; stock is enough local evidence for v1.
            if best.get("stock") in {None, ""}:
                return EvidenceGateDecision(True, "local result is missing stock/shipping evidence", best_score)
        return EvidenceGateDecision(False, "local evidence is sufficient", best_score)

    def _specific_product_terms(self, terms: list[str]) -> list[str]:
        generic = {
            "precio",
            "price",
            "envio",
            "envío",
            "shipping",
            "sensor",
            "component",
            "componente",
            "comprar",
            "buy",
        }
        specific = []
        for term in terms:
            normalized = term.strip().lower()
            if not normalized or normalized in generic:
                continue
            if re.search(r"\d", normalized) or "-" in normalized:
                specific.append(normalized)
        return specific

    def _best_result_contains_any(self, local: HybridSearchResponse, terms: list[str]) -> bool:
        if not local.results:
            return False
        best = local.results[0]
        haystack = " ".join(
            [
                str(best.item.get("sku", "")),
                str(best.item.get("name", "")),
                str(best.item.get("category", "")),
                str(best.item.get("description", "")),
                str(best.item.get("recommended_use", "")),
                str(best.document),
                " ".join(best.matched_terms),
            ]
        ).lower()
        return any(term in haystack for term in terms)
