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


st.set_page_config(page_title="Asesor multiagente de componentes", layout="wide")

st.title("Asesor multiagente de componentes electronicos")
st.caption("POC con agente RAG de catalogo, agente web, modelo de intencion, seguridad, metricas y auditoria.")

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
    include_delivery_details = st.checkbox("Incluir entrega/stock", value=False)
    st.divider()
    st.write("Ejemplos")
    st.code("Necesito medir temperatura y humedad con un ESP32")
    st.code("Que componentes uso para alimentar sensores de 3.3 V desde 5 V?")

query = st.text_area(
    "Consulta",
    value="Necesito un sensor ambiental para un prototipo IoT con ESP32 y alimentacion de 3.3 V",
    height=110,
)

if st.button("Consultar agentes", type="primary"):
    advisor = build_component_advisor()
    preferred_sites = [site.strip() for site in preferred_sites_text.split(",") if site.strip()]
    response = advisor.answer(
        query,
        top_k=top_k,
        web_limit=web_limit,
        web_method=web_method,
        preferred_sites=preferred_sites or None,
        preferred_only=preferred_only,
        include_delivery_details=include_delivery_details,
    )

    if not response.ok:
        st.error(response.final_answer)
    else:
        st.subheader("Respuesta final")
        st.write(response.final_answer)

        left, middle, right = st.columns(3)
        with left:
            st.metric("Intencion", response.intent.label if response.intent else "n/a")
            st.caption(f"Proveedor: {response.intent.provider if response.intent else 'n/a'}")
        with middle:
            st.metric("Hits RAG", response.metrics["counters"].get("catalog_hits", 0))
        with right:
            st.metric("Resultados web", response.metrics["counters"].get("web_results", 0))

        tab_catalog, tab_web, tab_metrics, tab_audit = st.tabs(["Catalogo", "Web", "Metricas", "Auditoria"])
        with tab_catalog:
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
