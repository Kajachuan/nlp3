# POC multiagente: asesor de componentes electronicos

## Objetivo

El sistema recibe una consulta tecnica sobre componentes electronicos y coordina dos agentes:

- `CatalogAgent`: consulta un RAG local sobre un catalogo JSON de componentes.
- `WebResearchAgent`: consulta una tool compatible con `web_search_tool` y usa un fallback offline para demo.

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

El adapter busca una funcion `search(query, limit)` en estos modulos:

- `web_search_tool`
- `web_search_tool.search`
- `web_search_tool.web_search`

Si la libreria real expone otro nombre, solo hay que ajustar `src/tools/external/web_search_adapter.py`.

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
