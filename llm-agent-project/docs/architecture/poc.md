# POC multiagente: asesor de componentes electronicos

## Objetivo

El sistema recibe una consulta tecnica sobre componentes electronicos y coordina dos agentes:

- `CatalogAgent`: consulta un RAG local sobre un catalogo JSON de componentes.
- `WebResearchAgent`: consulta la tool real `web-search-tool` via `WebSearchClient.search(...)` y usa un fallback offline solo cuando faltan credenciales o falla el proveedor.

El orquestador combina ambas respuestas, ejecuta un modelo preexistente opcional para clasificar intencion y registra la accion final en un log de auditoria.

## Como cubre la consigna

- API funcional: interfaz Streamlit en `app/main.py`.
- Dos agentes inteligentes: `CatalogAgent` y `WebResearchAgent`, coordinados por `ComponentAdvisorOrchestrator`.
- Comunicacion dinamica: el orquestador pasa la consulta validada, recoge respuestas y autoevaluaciones, y compone una respuesta final.
- RAG: `CatalogRetriever` consulta una coleccion persistente de Chroma en `data/processed/chroma`, construida desde `data/raw/electronic_components_catalog.json`.
- Seguridad: `validate_user_query` filtra entradas vacias, largas, secretos, prompt injection basica y payloads peligrosos.
- Modelo preexistente: `PretrainedIntentModel` intenta usar `facebook/bart-large-mnli` via `transformers`; si no esta disponible, usa fallback deterministicamente testeable.
- Flujo modular: agentes, tool adapters, politicas, telemetry y pipeline factory estan desacoplados.
- Costos y latencia: limites `top_k` y `web_limit`, cache de construccion del pipeline y fallback offline sin llamadas externas.
- Accion final automatica: cada respuesta se registra en `logs/advisor_audit.jsonl`.
- Evaluacion y metricas: tiempos por modulo, hits RAG, resultados web, confianza de intencion y precision@k proxy.

## Integracion de web_search_tool

El adapter usa el paquete real:

- Repo: `https://github.com/lsaraco/web-search-tool`
- Import: `from web_search_tool import WebSearchClient`
- Metodo: `WebSearchClient.search(query=..., method=..., limit=...)`

Backends soportados desde la UI:

- `tavily`: requiere `TAVILY_API_KEY`.
- `google_shopping`: requiere `SERPAPI_API_KEY`.
- `google_search`: requiere `SERPAPI_API_KEY` y `ZYTE_API_KEY`.

Crear un `.env` en `llm-agent-project` a partir de `.env.example`. La app carga ese archivo al construir el pipeline. Si no hay credenciales, el agente web muestra `offline-demo: missing ...` para que el resto de la demo siga funcionando.

## Ejecucion

```powershell
cd llm-agent-project
pip install -r requirements.txt
python scripts/ingest_catalog_to_chroma.py
streamlit run app/main.py
```

La app tambien indexa el catalogo automaticamente al arrancar si la coleccion no esta construida.

## Tests

```powershell
cd llm-agent-project
python -m pytest tests/unit
```
