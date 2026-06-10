from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from langgraph.graph import END, StateGraph

from src.agents.cores.catalog_agent import CatalogAgentResult
from src.agents.cores.local_hybrid_agent import LocalHybridAgentResult, LocalHybridSearchAgent
from src.agents.cores.web_research_agent import WebResearchAgent, WebResearchResult
from src.agents.orchestration.evidence_gate import EvidenceGateDecision, LocalEvidenceGate
from src.common.telemetry.langfuse_tracer import LangfuseTracer
from src.common.telemetry.metrics import Metrics, append_audit_event
from src.policies.security import clamp_int, validate_user_query
from src.rag.retrievers.catalog_retriever import CatalogHit
from src.runtime.config import LLMModelSelection, default_model_selection
from src.runtime.prompts import load_prompt
from src.runtime.providers.groq_llm import GroqLLMProvider, parse_json_object
from src.tools.external.web_search_adapter import WebSearchMethod
from src.tools.external.telegram_tool import send_telegram_message


@dataclass(frozen=True)
class PlannerDecision:
    route: str
    normalized_query: str
    product_terms: list[str]
    web_search_query: str
    needs_price: bool
    stream_message: str
    reason: str
    model: str
    provider: str


@dataclass(frozen=True)
class AggregatorDecision:
    should_answer: bool
    product_name: str | None
    price: str | None
    purchase_url: str | None
    vendor: str | None
    source: str
    confidence: float
    answer: str
    limitations: list[str]
    citations: list[str]
    model: str
    provider: str


@dataclass(frozen=True)
class AdvisorResponse:
    ok: bool
    final_answer: str
    catalog: CatalogAgentResult | None
    web: WebResearchResult | None
    intent: Any | None
    metrics: dict
    audit_log_path: Path
    planner_decision: PlannerDecision | None = None
    local: LocalHybridAgentResult | None = None
    evidence_gate: EvidenceGateDecision | None = None
    aggregator_decision: AggregatorDecision | None = None
    stream_messages: list[str] | None = None


class ComponentAdvisorOrchestrator:
    def __init__(
        self,
        local_hybrid_agent: LocalHybridSearchAgent,
        web_agent: WebResearchAgent,
        evidence_gate: LocalEvidenceGate,
        audit_log_path: Path,
        llm_provider: GroqLLMProvider | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self.local_hybrid_agent = local_hybrid_agent
        self.web_agent = web_agent
        self.evidence_gate = evidence_gate
        self.audit_log_path = audit_log_path
        self.llm_provider = llm_provider or GroqLLMProvider()
        self.tracer = tracer or LangfuseTracer()

    def answer(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 3,
        web_limit: int = 3,
        web_method: WebSearchMethod | None = None,
        preferred_sites: list[str] | None = None,
        preferred_only: bool = False,
        enable_telegram_notification: bool = False,
        model_selection: LLMModelSelection | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AdvisorResponse:
        models = model_selection or default_model_selection()
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
                stream_messages=[],
            )

        top_k = clamp_int(top_k, 1, 5)
        web_limit = clamp_int(web_limit, 1, 5)
        metrics.set_label("planner_model", models.planner_model)
        metrics.set_label("direct_response_model", models.direct_response_model)
        metrics.set_label("verifier_model", models.verifier_model)
        metrics.set_label("estimated_cost_usd", 0.0)
        metrics.set_label("langfuse_enabled", self.tracer.enabled)

        with self.tracer.trace("component_purchase_advisor", {"query": validation.sanitized}) as trace:
            graph = self._build_graph(
                metrics=metrics,
                trace=trace,
                models=models,
                top_k=top_k,
                web_limit=web_limit,
                web_method=web_method,
                preferred_sites=preferred_sites,
                preferred_only=preferred_only,
                enable_telegram_notification=enable_telegram_notification,
                progress_callback=progress_callback,
            )
            state = graph.invoke(
                {
                    "query": validation.sanitized,
                    "chat_history": chat_history or [],
                    "stream_messages": [],
                    "web": None,
                    "local": None,
                    "gate": None,
                    "aggregator": None,
                    "final_answer": "",
                    "enable_telegram_notification": enable_telegram_notification,
                }
            )

        planner = state.get("planner")
        local = state.get("local")
        web = state.get("web")
        gate = state.get("gate")
        aggregator = state.get("aggregator")
        final_answer = self._clean_markdown_fences(state.get("final_answer") or "")
        catalog = self._catalog_result_from_local(local) if local else None
        response = AdvisorResponse(
            ok=True,
            final_answer=final_answer,
            catalog=catalog,
            web=web,
            intent=planner,
            metrics=metrics.as_dict(),
            audit_log_path=self.audit_log_path,
            planner_decision=planner,
            local=local,
            evidence_gate=gate,
            aggregator_decision=aggregator,
            stream_messages=state.get("stream_messages", []),
        )
        if self.tracer.last_error:
            metrics.set_label("langfuse_error", self.tracer.last_error)
            response = AdvisorResponse(
                ok=response.ok,
                final_answer=response.final_answer,
                catalog=response.catalog,
                web=response.web,
                intent=response.intent,
                metrics=metrics.as_dict(),
                audit_log_path=response.audit_log_path,
                planner_decision=response.planner_decision,
                local=response.local,
                evidence_gate=response.evidence_gate,
                aggregator_decision=response.aggregator_decision,
                stream_messages=response.stream_messages,
            )
        self._audit(validation.sanitized, response)
        return response

    def _build_graph(
        self,
        metrics: Metrics,
        trace,
        models: LLMModelSelection,
        top_k: int,
        web_limit: int,
        web_method: WebSearchMethod | None,
        preferred_sites: list[str] | None,
        preferred_only: bool,
        enable_telegram_notification: bool,
        progress_callback: Callable[[str], None] | None = None,
    ):
        graph = StateGraph(dict)

        def emit(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        def planner_node(state: dict[str, Any]) -> dict[str, Any]:
            emit("Planificando la consulta con el LLM...")
            query = state["query"]
            with trace.span("planner", {"model": models.planner_model}):
                with metrics.timer("planner"):
                    planner = self._run_planner(
                        query,
                        models.planner_model,
                        trace,
                        metrics,
                        state.get("chat_history", []),
                    )
            emit(f"Ruta elegida: {planner.route}. Query web preparada: {planner.web_search_query}")
            metrics.set_label("route", planner.route)
            metrics.set_label("web_search_query", planner.web_search_query)
            messages = list(state.get("stream_messages", []))
            if planner.stream_message:
                messages.append(planner.stream_message)
            return {**state, "planner": planner, "stream_messages": messages}

        def direct_response_node(state: dict[str, Any]) -> dict[str, Any]:
            emit("Generando respuesta directa...")
            planner = state["planner"]
            with trace.span("direct_response", {"model": models.direct_response_model}):
                with metrics.timer("direct_response"):
                    final_answer = self._run_direct_response(
                            state["query"],
                            planner,
                        models.direct_response_model,
                        trace,
                        metrics,
                        state.get("chat_history", []),
                    )
            emit("Respuesta directa generada.")
            metrics.set_label("web_fallback_used", False)
            return {**state, "final_answer": final_answer}

        def local_hybrid_search_node(state: dict[str, Any]) -> dict[str, Any]:
            planner = state["planner"]
            messages = list(state.get("stream_messages", []))
            messages.append("Searching the local catalog with semantic and keyword retrieval.")
            with trace.span("local_hybrid_search"):
                with metrics.timer("local_hybrid_search"):
                    emit("Llamando RAG semantico sobre el catalogo local...")
                    emit("Haciendo keyword/BM25 search sobre SKU, nombre y descripcion...")
                    local = self.local_hybrid_agent.run(planner.normalized_query or state["query"], top_k=top_k)
            emit(
                "Busqueda local terminada: "
                f"RAG={len(local.response.rag_hits)}, "
                f"Keyword/BM25={len(local.response.keyword_hits)}, "
                f"Hybrid={len(local.response.results)}."
            )
            metrics.increment("rag_result_count", len(local.response.rag_hits))
            metrics.increment("keyword_result_count", len(local.response.keyword_hits))
            metrics.increment("local_result_count", len(local.response.results))
            metrics.timings_ms.update(local.response.timings_ms)
            if local.response.results:
                metrics.set_score("best_local_score", local.response.results[0].score)
            return {**state, "local": local, "stream_messages": messages}

        def evidence_gate_node(state: dict[str, Any]) -> dict[str, Any]:
            emit("Evaluando si la evidencia local alcanza o si hace falta fallback web...")
            planner = state["planner"]
            local = state["local"]
            with trace.span("local_evidence_gate"):
                gate = self.evidence_gate.decide(
                    local.response,
                    planner.needs_price,
                    required_terms=planner.product_terms,
                )
            metrics.set_label("web_fallback_used", gate.use_web_fallback)
            metrics.set_label("evidence_gate_reason", gate.reason)
            emit(f"Decision de evidencia: {gate.reason}. Fallback web={'si' if gate.use_web_fallback else 'no'}.")
            return {**state, "gate": gate}

        def web_fallback_node(state: dict[str, Any]) -> dict[str, Any]:
            planner = state["planner"]
            messages = list(state.get("stream_messages", []))
            messages.append("Local evidence is insufficient. Searching external suppliers with Google Shopping.")
            emit(f"Llamando web search / Google Shopping con query: {planner.web_search_query}")
            with trace.span("web_fallback", {"query": planner.web_search_query}):
                with metrics.timer("web_fallback"):
                    web = self.web_agent.run(
                        planner.web_search_query or planner.normalized_query,
                        limit=web_limit,
                        method=web_method,
                        preferred_sites=preferred_sites,
                        preferred_only=preferred_only,
                    )
            emit(f"Web search terminado: {len(web.results)} resultados.")
            metrics.increment("web_result_count", len(web.results))
            metrics.set_label("web_tool_mode", web.tool_mode)
            return {**state, "web": web, "stream_messages": messages}

        def aggregator_verifier_node(state: dict[str, Any]) -> dict[str, Any]:
            emit("Agregando resultados y verificando respuesta final con el LLM...")
            web = state.get("web")
            if web is None:
                metrics.increment("web_result_count", 0)
            with trace.span("aggregator_verifier", {"model": models.verifier_model}):
                with metrics.timer("aggregator_verifier"):
                    aggregator = self._run_aggregator(
                        state["query"],
                        state["planner"],
                        state["local"],
                        web,
                        state.get("gate"),
                        models.verifier_model,
                        trace,
                        metrics,
                        state.get("chat_history", []),
                    )
            emit("Respuesta final verificada.")
            final_answer = aggregator.answer
            if not aggregator.should_answer:
                final_answer = (
                    aggregator.answer
                    or "No encontre evidencia suficiente para recomendar un componente con precio y URL de compra confiables."
                )
            return {**state, "aggregator": aggregator, "final_answer": final_answer}

        def telegram_notification_node(state: dict[str, Any]) -> dict[str, Any]:
            aggregator = state.get("aggregator")
            if not state.get("enable_telegram_notification", False):
                metrics.set_label("telegram_message_sent", False)
                metrics.set_label("telegram_message_skipped_reason", "disabled by user")
                return {**state, "telegram_result": None}
            if not self._should_send_telegram_message(aggregator):
                metrics.set_label("telegram_message_sent", False)
                metrics.set_label("telegram_message_skipped_reason", "missing purchase evidence")
                return {**state, "telegram_result": None}

            emit("Enviando mensaje de compra por Telegram...")
            with trace.span("telegram_notification"):
                with metrics.timer("telegram_notification"):
                    telegram_result = send_telegram_message.invoke(
                        {
                            "product_name": aggregator.product_name,
                            "price": aggregator.price,
                            "link": aggregator.purchase_url,
                        }
                    )
            sent = bool(telegram_result.get("ok"))
            metrics.set_label("telegram_message_sent", sent)
            metrics.set_label("telegram_chat_id", str(telegram_result.get("chat_id") or ""))
            if not sent:
                metrics.set_label("telegram_message_error", str(telegram_result.get("error") or "unknown error"))
                emit("No se pudo enviar el mensaje por Telegram. La respuesta final se mantiene.")
            else:
                emit("Mensaje de compra enviado por Telegram.")
            return {**state, "telegram_result": telegram_result}

        graph.add_node("planner", planner_node)
        graph.add_node("direct_response", direct_response_node)
        graph.add_node("local_hybrid_search", local_hybrid_search_node)
        graph.add_node("local_evidence_gate", evidence_gate_node)
        graph.add_node("web_fallback", web_fallback_node)
        graph.add_node("aggregator_verifier", aggregator_verifier_node)
        graph.add_node("telegram_notification", telegram_notification_node)
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            lambda state: "direct_response" if state["planner"].route in {"out_of_domain", "agent_info"} else "local_hybrid_search",
            {
                "direct_response": "direct_response",
                "local_hybrid_search": "local_hybrid_search",
            },
        )
        graph.add_edge("direct_response", END)
        graph.add_edge("local_hybrid_search", "local_evidence_gate")
        graph.add_conditional_edges(
            "local_evidence_gate",
            lambda state: "web_fallback" if state["gate"].use_web_fallback else "aggregator_verifier",
            {
                "web_fallback": "web_fallback",
                "aggregator_verifier": "aggregator_verifier",
            },
        )
        graph.add_edge("web_fallback", "aggregator_verifier")
        graph.add_conditional_edges(
            "aggregator_verifier",
            lambda state: "telegram_notification"
            if enable_telegram_notification and self._should_send_telegram_message(state.get("aggregator"))
            else "end",
            {
                "telegram_notification": "telegram_notification",
                "end": END,
            },
        )
        graph.add_edge("telegram_notification", END)
        return graph.compile()

    def _run_planner(
        self,
        query: str,
        model: str,
        trace,
        metrics: Metrics,
        chat_history: list[dict[str, str]] | None = None,
    ) -> PlannerDecision:
        prompt = load_prompt("planner")
        payload = self._conversation_payload(query, chat_history)
        llm_result = self.llm_provider.invoke(model, prompt, payload)
        self._record_llm_usage(metrics, "planner", llm_result)
        trace.generation("planner", model, payload, llm_result.content, {"provider": llm_result.provider})
        parsed = parse_json_object(llm_result.content)
        if not parsed or "error" in parsed:
            parsed = self._fallback_planner(query, model, llm_result.provider)
        return PlannerDecision(
            route=str(parsed.get("route") or "local_search"),
            normalized_query=str(parsed.get("normalized_query") or query),
            product_terms=[str(term) for term in parsed.get("product_terms", []) if str(term).strip()],
            web_search_query=str(parsed.get("web_search_query") or self._english_web_query(query)),
            needs_price=bool(parsed.get("needs_price", True)),
            stream_message=str(parsed.get("stream_message") or "I am planning the best way to answer your request."),
            reason=str(parsed.get("reason") or ""),
            model=model,
            provider=llm_result.provider,
        )

    def _run_direct_response(
        self,
        query: str,
        planner: PlannerDecision,
        model: str,
        trace,
        metrics: Metrics,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        prompt = load_prompt("direct_response")
        payload = (
            f"{self._conversation_payload(query, chat_history)}\n"
            f"Route: {planner.route}\nReason: {planner.reason}"
        )
        llm_result = self.llm_provider.invoke(model, prompt, payload)
        self._record_llm_usage(metrics, "direct_response", llm_result)
        trace.generation("direct_response", model, payload, llm_result.content, {"provider": llm_result.provider})
        if llm_result.provider == "groq" and llm_result.content.strip():
            return self._clean_markdown_fences(llm_result.content)
        if planner.route == "agent_info":
            return (
                "Este agente busca productos electronicos para comprar. Primero consulta nuestro catalogo local y, "
                "si no alcanza, usa proveedores externos para encontrar precio y URL de compra."
            )
        return (
            "No puedo responder esa consulta porque este agente esta enfocado en buscar productos electronicos "
            "para compra, precio y URL de compra."
        )

    def _run_aggregator(
        self,
        query: str,
        planner: PlannerDecision,
        local: LocalHybridAgentResult,
        web: WebResearchResult | None,
        gate: EvidenceGateDecision | None,
        model: str,
        trace,
        metrics: Metrics,
        chat_history: list[dict[str, str]] | None = None,
    ) -> AggregatorDecision:
        prompt = load_prompt("aggregator_verifier")
        local_results_for_aggregation = local.response.results
        if web and web.results and not self._local_supports_specific_terms(local, planner):
            local_results_for_aggregation = []
            metrics.set_label("local_results_suppressed_for_aggregation", True)
        else:
            metrics.set_label("local_results_suppressed_for_aggregation", False)

        payload = {
            "query": query,
            "chat_history": chat_history or [],
            "planner": planner.__dict__,
            "evidence_gate": gate.__dict__ if gate else None,
            "aggregation_rule": (
                "If local_results is empty and web_results is not empty, answer from web_results. "
                "Do not use omitted local candidates."
            ),
            "local_results": [self._hybrid_to_dict(result) for result in local_results_for_aggregation],
            "web_results": [result.__dict__ for result in web.results] if web else [],
        }
        payload_text = str(payload)
        llm_result = self.llm_provider.invoke(model, prompt, payload_text)
        self._record_llm_usage(metrics, "aggregator_verifier", llm_result)
        trace.generation("aggregator_verifier", model, payload_text, llm_result.content, {"provider": llm_result.provider})
        parsed = parse_json_object(llm_result.content)
        if parsed and "error" not in parsed and parsed.get("answer"):
            parsed_purchase_url = self._optional_str(parsed.get("purchase_url") or parsed.get("source"))
            if not parsed_purchase_url and local_results_for_aggregation:
                parsed_purchase_url = self._local_purchase_url(local_results_for_aggregation[0].item)
            return AggregatorDecision(
                should_answer=bool(parsed.get("should_answer", True)),
                product_name=self._optional_str(parsed.get("product_name")),
                price=self._optional_str(parsed.get("price")),
                purchase_url=parsed_purchase_url,
                vendor=self._optional_str(parsed.get("vendor")),
                source=str(parsed.get("source") or parsed_purchase_url or ""),
                confidence=float(parsed.get("confidence") or 0.0),
                answer=self._clean_markdown_fences(str(parsed.get("answer") or "")),
                limitations=[str(item) for item in parsed.get("limitations", [])],
                citations=[str(item) for item in parsed.get("citations", [])],
                model=model,
                provider=llm_result.provider,
            )
        return self._fallback_aggregator(local, web, planner, model, llm_result.provider)

    def _fallback_planner(self, query: str, model: str, provider: str) -> dict[str, Any]:
        lowered = query.lower()
        agent_terms = ["que haces", "qué haces", "como funciona", "cómo funciona", "help", "ayuda", "alcance"]
        product_terms = ["precio", "price", "comprar", "buy", "stock", "url", "link", "sku", "sensor", "op amp", "amplificador", "regulador", "lm741", "esp32", "bme280"]
        if any(term in lowered for term in agent_terms):
            route = "agent_info"
        elif not any(term in lowered for term in product_terms):
            route = "out_of_domain"
        else:
            route = "local_search"
        return {
            "route": route,
            "normalized_query": query,
            "product_terms": self._product_terms(query),
            "web_search_query": self._english_web_query(query),
            "needs_price": any(term in lowered for term in ["precio", "price", "comprar", "buy"]),
            "stream_message": "I am checking whether this should use local catalog search or external suppliers.",
            "reason": f"fallback planner via {provider}",
        }

    def _fallback_aggregator(
        self,
        local: LocalHybridAgentResult,
        web: WebResearchResult | None,
        planner: PlannerDecision,
        model: str,
        provider: str,
    ) -> AggregatorDecision:
        local_is_supported = self._local_supports_specific_terms(local, planner)
        if web and web.results and not local_is_supported:
            best_web = web.results[0]
            answer = (
                f"No encontre una coincidencia suficiente en el catalogo local. "
                f"Resultado externo: {best_web.title}.\n\n"
                f"Precio: {best_web.price or 'No informado'}\n"
                f"Comprar: [Click aqui]({self._markdown_url(best_web.url)})"
            )
            return AggregatorDecision(
                should_answer=True,
                product_name=best_web.title,
                price=best_web.price,
                purchase_url=best_web.url,
                vendor=best_web.vendor or best_web.source,
                source=best_web.url,
                confidence=best_web.score or 0.5,
                answer=answer,
                limitations=["External result should be verified before purchase."],
                citations=[best_web.url],
                model=model,
                provider=provider,
            )

        if local.response.results and local_is_supported:
            best = local.response.results[0]
            item = best.item
            purchase_url = self._local_purchase_url(item)
            answer = (
                f"En nuestro catalogo encontre {item['name']} ({item['sku']}).\n\n"
                f"Precio: USD {item.get('price_usd')}\n"
                f"Comprar: [Click aqui]({self._markdown_url(purchase_url)})"
            )
            return AggregatorDecision(
                should_answer=True,
                product_name=str(item.get("name")),
                price=f"USD {item.get('price_usd')}",
                purchase_url=purchase_url,
                vendor="catalog",
                source=purchase_url,
                confidence=min(1.0, max(0.0, best.score)),
                answer=answer,
                limitations=[],
                citations=[purchase_url],
                model=model,
                provider=provider,
            )
        if web and web.results:
            best_web = web.results[0]
            answer = (
                f"No encontre una coincidencia suficiente en el catalogo local. "
                f"Resultado externo: {best_web.title}.\n\n"
                f"Precio: {best_web.price or 'No informado'}\n"
                f"Comprar: [Click aqui]({self._markdown_url(best_web.url)})"
            )
            return AggregatorDecision(
                should_answer=True,
                product_name=best_web.title,
                price=best_web.price,
                purchase_url=best_web.url,
                vendor=best_web.vendor or best_web.source,
                source=best_web.url,
                confidence=best_web.score or 0.5,
                answer=answer,
                limitations=["External result should be verified before purchase."],
                citations=[best_web.url],
                model=model,
                provider=provider,
            )
        return AggregatorDecision(
            should_answer=False,
            product_name=None,
            price=None,
            purchase_url=None,
            vendor=None,
            source="none",
            confidence=0.0,
            answer="No encontre evidencia suficiente para recomendar un componente con precio y URL de compra confiables.",
            limitations=["No local or external evidence available."],
            citations=[],
            model=model,
            provider=provider,
        )

    def _catalog_result_from_local(self, local: LocalHybridAgentResult) -> CatalogAgentResult:
        hits = [
            CatalogHit(item=result.item, score=result.semantic_score or result.score, matched_terms=result.matched_terms)
            for result in local.response.results
        ]
        return CatalogAgentResult(answer=local.answer, hits=hits, self_review=local.self_review)

    def _hybrid_to_dict(self, result) -> dict[str, Any]:
        return {
            "item": result.item,
            "purchase_url": self._local_purchase_url(result.item),
            "score": result.score,
            "semantic_score": result.semantic_score,
            "keyword_score": result.keyword_score,
            "matched_terms": result.matched_terms,
            "sources": result.sources,
            "document": result.document,
        }

    def _product_terms(self, query: str) -> list[str]:
        return [term for term in re.findall(r"[A-Za-z0-9.-]+", query) if len(term) > 2]

    def _local_supports_specific_terms(self, local: LocalHybridAgentResult, planner: PlannerDecision) -> bool:
        specific_terms = self._specific_product_terms(
            [
                *planner.product_terms,
                *self._product_terms(planner.normalized_query),
                *self._product_terms(planner.web_search_query),
            ]
        )
        if not specific_terms:
            return bool(local.response.results)
        for result in local.response.results:
            haystack = " ".join(
                [
                    str(result.item.get("sku", "")),
                    str(result.item.get("name", "")),
                    str(result.item.get("category", "")),
                    str(result.item.get("description", "")),
                    str(result.item.get("recommended_use", "")),
                    str(result.document),
                    " ".join(result.matched_terms),
                ]
            ).lower()
            if any(term in haystack for term in specific_terms):
                return True
        return False

    def _specific_product_terms(self, terms: list[str]) -> list[str]:
        generic = {
            "precio",
            "price",
            "buy",
            "comprar",
            "stock",
            "url",
            "link",
            "component",
            "componente",
            "allegro",
            "sensor",
        }
        specific = []
        for term in terms:
            normalized = term.strip().lower()
            if not normalized or normalized in generic:
                continue
            if re.search(r"\d", normalized) or "-" in normalized:
                specific.append(normalized)
        return sorted(set(specific))

    def _english_web_query(self, query: str) -> str:
        lowered = query.lower()
        replacements = {
            "precio": "",
            "dame": "",
            "quiero": "",
            "necesito": "",
            "amplificador operacional": "op amp",
            "bajo ruido": "low noise",
            "canales": "channel",
            "canal": "channel",
            "de": "",
            "un": "",
            "una": "",
            "el": "",
            "la": "",
        }
        if "amplificador operacional" in lowered and "4" in lowered:
            return "low noise 4 channel op amp" if "bajo ruido" in lowered else "4 channel op amp"
        text = lowered
        for source, target in replacements.items():
            text = text.replace(source, target)
        tokens = [token for token in re.findall(r"[a-zA-Z0-9.-]+", text) if len(token) > 1]
        return " ".join(tokens).strip() or query

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return None if text.lower() == "null" or not text.strip() else text

    def _should_send_telegram_message(self, aggregator: AggregatorDecision | None) -> bool:
        """Return whether the final purchase answer has enough data for Telegram."""
        if not aggregator or not aggregator.should_answer:
            return False
        return bool(aggregator.product_name and aggregator.price and aggregator.purchase_url)

    def _clean_markdown_fences(self, text: str) -> str:
        cleaned = text.strip()
        fenced = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()
        return self._normalize_markdown_links(cleaned)

    def _normalize_markdown_links(self, text: str) -> str:
        """Percent-encode unsafe characters inside Markdown link URLs."""

        def replace(match: re.Match[str]) -> str:
            label = match.group("label")
            url = match.group("url").strip()
            return f"[{label}]({self._markdown_url(url)})"

        return re.sub(r"\[(?P<label>[^\]]+)\]\((?P<url>[^\n]+)\)", replace, text)

    def _markdown_url(self, url: str) -> str:
        """Return a URL safe for Markdown link destinations."""
        return quote(str(url).strip(), safe=":/?#[]@!$&'*+,;=%")

    def _local_purchase_url(self, item: dict[str, Any]) -> str:
        component_name = str(item.get("name") or item.get("sku") or "component")
        base_url = os.getenv(
            "LOCAL_COMPONENT_PAGE_BASE_URL",
            "http://localhost:5500/llm-agent-project/web/componentes.html",
        ).rstrip("?&")
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}id={quote(component_name, safe='')}"

    def _record_llm_usage(self, metrics: Metrics, node_name: str, llm_result) -> None:
        input_tokens = int(llm_result.prompt_tokens or 0)
        output_tokens = int(llm_result.completion_tokens or 0)
        total_tokens = input_tokens + output_tokens
        metrics.increment("input_tokens", input_tokens)
        metrics.increment("output_tokens", output_tokens)
        metrics.increment("total_tokens", total_tokens)
        metrics.increment(f"{node_name}_input_tokens", input_tokens)
        metrics.increment(f"{node_name}_output_tokens", output_tokens)
        metrics.increment(f"{node_name}_total_tokens", total_tokens)
        current_cost = float(metrics.labels.get("estimated_cost_usd", 0.0) or 0.0)
        metrics.set_label("estimated_cost_usd", round(current_cost + float(llm_result.estimated_cost or 0.0), 6))

    def _conversation_payload(self, query: str, chat_history: list[dict[str, str]] | None = None) -> str:
        history = chat_history or []
        recent = history[-6:]
        lines = ["Recent conversation:"]
        if recent:
            for message in recent:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                lines.append(f"- {role}: {content}")
        else:
            lines.append("- none")
        lines.append(f"Current user query: {query}")
        return "\n".join(lines)

    def _audit(self, query: str, response: AdvisorResponse) -> None:
        metrics = response.metrics
        append_audit_event(
            self.audit_log_path,
            {
                "event": "advisor_response",
                "query": query,
                "route": response.planner_decision.route if response.planner_decision else None,
                "web_search_query": response.planner_decision.web_search_query if response.planner_decision else None,
                "web_fallback_used": metrics.get("labels", {}).get("web_fallback_used"),
                "metrics": metrics,
            },
        )
