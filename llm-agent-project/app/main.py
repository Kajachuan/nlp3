from __future__ import annotations

import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.runtime.pipelines.factory import build_component_advisor
from src.runtime.config import LLMModelSelection, default_model_selection, parse_model_options


st.set_page_config(page_title="Agente de compra de componentes", layout="wide")

st.title("Agente de compra de componentes electronicos")
st.caption("POC con planner LLM, busqueda local hibrida, fallback Google Shopping, Langfuse, metricas y auditoria.")

web_methods = ["tavily", "google_shopping", "google_search"]
default_web_method = os.getenv("WEB_SEARCH_DEFAULT_METHOD")
if not default_web_method and os.getenv("SERPAPI_API_KEY"):
    default_web_method = "google_shopping"
if default_web_method not in web_methods:
    default_web_method = "tavily"

with st.sidebar:
    st.header("Configuracion")
    top_k = st.slider("Resultados RAG", min_value=1, max_value=5, value=3)
    web_limit = st.slider("Resultados web", min_value=1, max_value=5, value=3)
    web_method = st.selectbox("Metodo web", web_methods, index=web_methods.index(default_web_method))
    preferred_sites_text = st.text_input("Sitios preferidos", value="digikey.com,mouser.com")
    preferred_only = st.checkbox("Solo sitios preferidos", value=False)
    st.divider()
    st.subheader("Modelos LLM")
    model_options = parse_model_options()
    defaults = default_model_selection()
    planner_model = st.selectbox(
        "Planner",
        model_options,
        index=model_options.index(defaults.planner_model) if defaults.planner_model in model_options else 0,
    )
    direct_response_model = st.selectbox(
        "Respuesta directa",
        model_options,
        index=model_options.index(defaults.direct_response_model) if defaults.direct_response_model in model_options else 0,
    )
    verifier_model = st.selectbox(
        "Agregador/verificador",
        model_options,
        index=model_options.index(defaults.verifier_model) if defaults.verifier_model in model_options else 0,
    )
    st.divider()
    st.write("Ejemplos")
    st.code("Dame el precio del BME280")
    st.code("Quiero comprar un amplificador operacional de bajo ruido")
    if st.button("Limpiar chat"):
        st.session_state["messages"] = []
        st.session_state.pop("last_response", None)
        st.rerun()


def render_table_dialog(title: str, rows: list[dict]) -> None:
    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dialog():
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No hay resultados para mostrar.")

        _dialog()
    else:
        with st.expander(title, expanded=True):
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No hay resultados para mostrar.")


def response_rows(response):
    rag_rows = []
    keyword_rows = []
    web_rows = []
    if response.local:
        rag_rows = [
            {
                "rank": index,
                "sku": hit.item.get("sku"),
                "name": hit.item.get("name"),
                "score": hit.score,
                "matched_terms": ", ".join(hit.matched_terms),
            }
            for index, hit in enumerate(response.local.response.rag_hits, start=1)
        ]
        keyword_rows = [
            {
                "rank": index,
                "sku": hit.item.get("sku"),
                "name": hit.item.get("name"),
                "score": hit.score,
                "matched_terms": ", ".join(hit.matched_terms),
                "text": hit.document,
            }
            for index, hit in enumerate(response.local.response.keyword_hits, start=1)
        ]
    if response.web:
        web_rows = [
            {
                "rank": index,
                "title": result.title,
                "vendor": result.vendor,
                "price": result.price,
                "score": result.score,
                "url": result.url,
                "snippet": result.snippet,
            }
            for index, result in enumerate(response.web.results, start=1)
        ]
    return rag_rows, keyword_rows, web_rows


def render_timing_bar(timings: dict[str, float]) -> None:
    if not timings:
        return
    display_timings = dict(timings)
    if {"rag_search", "keyword_bm25_search", "hybrid_merge"}.intersection(display_timings):
        display_timings.pop("local_hybrid_search", None)
    order = [
        "planner",
        "direct_response",
        "rag_search",
        "keyword_bm25_search",
        "hybrid_merge",
        "web_fallback",
        "aggregator_verifier",
        "telegram_notification",
    ]
    ordered_timings = {key: display_timings[key] for key in order if key in display_timings}
    for key, value in display_timings.items():
        if key not in ordered_timings:
            ordered_timings[key] = value

    total = sum(float(value) for value in ordered_timings.values())
    if total <= 0:
        return
    colors = {
        "planner": "#4f46e5",
        "direct_response": "#7c3aed",
        "rag_search": "#059669",
        "keyword_bm25_search": "#16a34a",
        "hybrid_merge": "#84cc16",
        "web_fallback": "#0284c7",
        "aggregator_verifier": "#dc2626",
        "telegram_notification": "#ea580c",
    }
    default_color = "#64748b"
    segments = []
    legend = []
    for name, value in ordered_timings.items():
        width = max((float(value) / total) * 100, 1.0)
        color = colors.get(name, default_color)
        label = name.replace("_", " ")
        segments.append(
            f"<div title='{label}: {value} ms' "
            f"style='width:{width:.2f}%;background:{color};height:18px;'></div>"
        )
        legend.append(
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;margin-top:6px;'>"
            f"<span style='width:10px;height:10px;background:{color};display:inline-block;border-radius:2px;'></span>"
            f"<span>{label}: {value} ms</span></span>"
        )
    st.markdown(
        "<div style='margin:10px 0 4px;font-weight:600;'>Tiempo por etapa</div>"
        "<div style='display:flex;width:100%;overflow:hidden;border-radius:6px;border:1px solid #d7dde8;'>"
        + "".join(segments)
        + "</div>"
        "<div style='font-size:13px;color:#5b6678;'>"
        + "".join(legend)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_response(response, key_prefix: str = "response", show_answer: bool = True) -> None:
    if not response.ok:
        st.error(response.final_answer)
        return

    if show_answer:
        st.subheader("Respuesta final")
        st.markdown(response.final_answer)
    if response.stream_messages:
        with st.expander("Mensajes de estado"):
            for message in response.stream_messages:
                st.write(message)

    timings = response.metrics.get("timings_ms", {})
    counters = response.metrics.get("counters", {})
    labels = response.metrics.get("labels", {})
    scores = response.metrics.get("scores", {})
    total_ms = round(sum(timings.values()), 2)

    cols = st.columns(5)
    cols[0].metric("Ruta", labels.get("route", "n/a"))
    cols[1].metric("Tiempo total", f"{total_ms} ms")
    cols[2].metric("Fallback web", "si" if labels.get("web_fallback_used") else "no")
    cols[3].metric("Mejor score local", scores.get("best_local_score", 0))
    cols[4].metric("Costo estimado", labels.get("estimated_cost_usd", "n/a"))
    token_cols = st.columns(3)
    token_cols[0].metric("Input tokens", counters.get("input_tokens", 0))
    token_cols[1].metric("Output tokens", counters.get("output_tokens", 0))
    token_cols[2].metric("Total tokens", counters.get("total_tokens", 0))

    render_timing_bar(timings)

    rag_rows, keyword_rows, web_rows = response_rows(response)
    result_cols = st.columns(3)
    if result_cols[0].button(
        f"Ver resultados RAG ({counters.get('rag_result_count', 0)})",
        key=f"{key_prefix}_rag_results",
    ):
        render_table_dialog("Resultados RAG", rag_rows)
    if result_cols[1].button(
        f"Ver resultados Keyword/BM25 ({counters.get('keyword_result_count', 0)})",
        key=f"{key_prefix}_keyword_results",
    ):
        render_table_dialog("Resultados Keyword/BM25", keyword_rows)
    if result_cols[2].button(
        f"Ver resultados Web ({counters.get('web_result_count', 0)})",
        key=f"{key_prefix}_web_results",
    ):
        render_table_dialog("Resultados Web", web_rows)

    tab_catalog, tab_web, tab_metrics, tab_audit = st.tabs(["Catalogo", "Web", "Metricas", "Auditoria"])
    with tab_catalog:
        if response.planner_decision:
            st.json(
                {
                    "route": response.planner_decision.route,
                    "normalized_query": response.planner_decision.normalized_query,
                    "web_search_query": response.planner_decision.web_search_query,
                    "reason": response.planner_decision.reason,
                }
            )
        if response.catalog:
            for hit in response.catalog.hits:
                item = hit.item
                st.write(f"**{item['name']}** - `{item['sku']}`")
                st.write(item["description"])
                st.json({"score": hit.score, "matched_terms": hit.matched_terms, "specs": item["specs"]})
    with tab_web:
        if response.web:
            st.caption(f"Modo de tool: {response.web.tool_mode}")
            for result in response.web.results:
                st.write(f"**{result.title}**")
                if result.vendor or result.price or result.score is not None:
                    st.json({"vendor": result.vendor, "price": result.price, "score": result.score})
                st.write(result.snippet)
                st.write(result.url)
    with tab_metrics:
        st.json(response.metrics)
    with tab_audit:
        st.write(f"Log generado en `{response.audit_log_path}`")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for index, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        response = message.get("response")
        if response:
            with st.expander("Detalles de la respuesta", expanded=False):
                render_response(response, key_prefix=f"message_{index}", show_answer=False)

prompt = st.chat_input("Escribi que componente queres buscar, comparar o comprar")

if prompt:
    prior_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state["messages"]
        if message.get("content")
    ]
    st.session_state["messages"].append({"role": "user", "content": prompt})

    advisor = build_component_advisor()
    preferred_sites = [site.strip() for site in preferred_sites_text.split(",") if site.strip()]
    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Recibi la consulta. Preparando agentes...")

        def update_status(message: str) -> None:
            status.info(message)

        response = advisor.answer(
            prompt,
            chat_history=prior_history,
            top_k=top_k,
            web_limit=web_limit,
            web_method=web_method,
            preferred_sites=preferred_sites or None,
            preferred_only=preferred_only,
            model_selection=LLMModelSelection(
                planner_model=planner_model,
                direct_response_model=direct_response_model,
                verifier_model=verifier_model,
            ),
            progress_callback=update_status,
        )
        status.empty()
        st.markdown(response.final_answer)
    st.session_state["last_response"] = response
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response.final_answer,
            "response": response,
        }
    )
    st.rerun()

if not st.session_state["messages"]:
    st.info("Empeza preguntando por precio o URL de compra de un componente electronico.")
