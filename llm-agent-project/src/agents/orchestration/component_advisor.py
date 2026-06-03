from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents.cores.catalog_agent import CatalogAgent, CatalogAgentResult
from src.agents.cores.web_research_agent import WebResearchAgent, WebResearchResult
from src.common.telemetry.metrics import Metrics, append_audit_event
from src.policies.security import clamp_int, validate_user_query
from src.runtime.providers.pretrained_intent_model import IntentPrediction, PretrainedIntentModel
from src.tools.external.web_search_adapter import WebSearchMethod


@dataclass(frozen=True)
class AdvisorResponse:
    ok: bool
    final_answer: str
    catalog: CatalogAgentResult | None
    web: WebResearchResult | None
    intent: IntentPrediction | None
    metrics: dict
    audit_log_path: Path


class ComponentAdvisorOrchestrator:
    def __init__(
        self,
        catalog_agent: CatalogAgent,
        web_agent: WebResearchAgent,
        intent_model: PretrainedIntentModel,
        audit_log_path: Path,
    ) -> None:
        self.catalog_agent = catalog_agent
        self.web_agent = web_agent
        self.intent_model = intent_model
        self.audit_log_path = audit_log_path

    def answer(
        self,
        query: str,
        top_k: int = 3,
        web_limit: int = 3,
        web_method: WebSearchMethod | None = None,
        preferred_sites: list[str] | None = None,
        preferred_only: bool = False,
        include_delivery_details: bool = False,
    ) -> AdvisorResponse:
        metrics = Metrics()
        validation = validate_user_query(query)
        if not validation.is_valid:
            metrics.increment("blocked_queries")
            append_audit_event(
                self.audit_log_path,
                {"event": "blocked_query", "reason": validation.reason, "query": validation.sanitized},
            )
            return AdvisorResponse(
                ok=False,
                final_answer=f"Consulta rechazada: {validation.reason}",
                catalog=None,
                web=None,
                intent=None,
                metrics=metrics.as_dict(),
                audit_log_path=self.audit_log_path,
            )

        top_k = clamp_int(top_k, 1, 5)
        web_limit = clamp_int(web_limit, 1, 5)

        with metrics.timer("intent_model"):
            intent = self.intent_model.predict(validation.sanitized)
        with metrics.timer("catalog_agent"):
            catalog = self.catalog_agent.run(validation.sanitized, top_k=top_k)
        with metrics.timer("web_agent"):
            web = self.web_agent.run(
                validation.sanitized,
                limit=web_limit,
                method=web_method,
                preferred_sites=preferred_sites,
                preferred_only=preferred_only,
                include_delivery_details=include_delivery_details,
            )

        metrics.increment("catalog_hits", len(catalog.hits))
        metrics.increment("web_results", len(web.results))
        metrics.set_score("intent_confidence", intent.confidence)
        if catalog.hits:
            metrics.set_score("rag_best_score", catalog.hits[0].score)
            metrics.set_score("rag_precision_at_k_proxy", self._precision_proxy(catalog.hits))

        final_answer = self._compose_answer(validation.sanitized, intent, catalog, web)
        append_audit_event(
            self.audit_log_path,
            {
                "event": "advisor_response",
                "query": validation.sanitized,
                "intent": intent.label,
                "catalog_hits": len(catalog.hits),
                "web_results": len(web.results),
                "tool_mode": web.tool_mode,
                "metrics": metrics.as_dict(),
            },
        )

        return AdvisorResponse(
            ok=True,
            final_answer=final_answer,
            catalog=catalog,
            web=web,
            intent=intent,
            metrics=metrics.as_dict(),
            audit_log_path=self.audit_log_path,
        )

    def _compose_answer(
        self,
        query: str,
        intent: IntentPrediction,
        catalog: CatalogAgentResult,
        web: WebResearchResult,
    ) -> str:
        best = self._select_recommendation(intent, catalog)
        if best:
            recommendation = (
                f"Para la consulta '{query}', recomiendo empezar por {best['name']} ({best['sku']}). "
                f"La intencion detectada fue {intent.label} con confianza {intent.confidence:.2f}. "
                f"El catalogo lo respalda por uso, disponibilidad y costo: stock {best['stock']} y precio aprox. USD {best['price_usd']}."
            )
        else:
            recommendation = (
                f"Para la consulta '{query}' no hay una recomendacion fuerte dentro del catalogo. "
                "Usaria la evidencia externa para reformular requisitos o ampliar componentes."
            )

        return "\n\n".join(
            [
                recommendation,
                "Agente RAG/catalogo:\n" + catalog.answer,
                "Agente web:\n" + web.answer,
                "Autoevaluacion:\n"
                f"- Catalogo: {catalog.self_review}\n"
                f"- Web: {web.self_review}\n"
                "- Accion final: se registro el evento en el log de auditoria.",
            ]
        )

    def _precision_proxy(self, hits) -> float:
        if not hits:
            return 0.0
        relevant = sum(1 for hit in hits if hit.score >= 0.04)
        return relevant / len(hits)

    def _select_recommendation(self, intent: IntentPrediction, catalog: CatalogAgentResult) -> dict | None:
        if not catalog.hits:
            return None

        preferred_category_by_intent = {
            "sensor_selection": "sensor",
            "power_design": "power",
            "microcontroller_selection": "microcontroller",
        }
        preferred_category = preferred_category_by_intent.get(intent.label)
        if preferred_category:
            for hit in catalog.hits:
                if hit.item.get("category") == preferred_category:
                    return hit.item
        return catalog.hits[0].item
