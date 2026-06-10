# Sobre el proyecto

Es para la materia NLP3 (Procesamiento del Lenguaje Natural 3) de la Maestria en IA de la UBA. Se trata de uso avanzados de LLM, agentes, etc.
En nuestro caso es un agente para compra de componentes electronicos (resistencias, capacitores, ICs, etc) en nuestro catalogo interno (mediante RAG) o con provedores exernos (mediante web search)

## Codigo

- Python 3.13 o mayor
- El workspace tiene su propio virtual environment `.venv` y todo codigo que corras tiene que correrse a traves del .venv, incluso los tests de pytest
- Funciones con docstring de una linea si son pequenas o mutlilinea google style si son complejas
- PEP-8 compatible y seguiendo estandares de buenas practicas
- Usa uv como package manager. Los paquetes se instalan con `uv pip install`. Asegurate de mantener requirements.txt actualizado

## Documentacion

- La documentacion viva de todo el proyecto esta en docs/architecture.html. Leyendo ese HTML uno podria saber todo lo que hace el proyecto
- El HTML se debe mantener visual, conciso y facil de digerir. Se priorizan los diagramas en bloque y diagramas de flujo de datos


## Consigna a cumplir

El proyecto debe cumplir con estos requisitos:

  **API funcional:** Implementar una interfaz mediante FastAPI (preferido por escalabilidad y flexibilidad) o Streamlit (para prototipos visuales), que permita orquestar las interacciones entre usuarios,
  agentes y modelos de lenguaje.

  **Al menos dos agentes inteligentes con comunicación dinámica:** Diseñar al menos dos agentes autónomos, cada uno con roles diferenciados (por ejemplo, analista y verificador), que interactúen mediante
  prompts estructurados, mensajes reflexivos o cadenas de razonamiento. Incorporar mecanismos de autoevaluación o retroalimentación cuando sea relevante (ej. agentes autoreflexivos o planificadores).

  **Recuperación de contexto con RAG** (Retrieval-Augmented Generation): Implementar un componente de recuperación semántica, y modelos tipo retriever-ranker. Asegurar la relevancia del contenido recuperado y
  la trazabilidad de fuentes.

  **Implementar medidas de seguridad:** Definir políticas de control de acceso, filtrado de inputs, validación de outputs y auditoría continua para garantizar robustez y confiabilidad del sistema, según
  corresponda.

  **Integración de modelo preexistente para inferencia:** Utilizar un modelo CNN/ViT ya entrenado (propio o alojado en plataformas como Hugging Face) para realizar tareas de inferencia.

  **Flujo de datos completo y modular:** Asegurar un pipeline de procesamiento claro, donde la entrada fluya a través de los distintos agentes, RAG y modelo, hasta generar una respuesta final. Incorporar
  componentes desacoplables y monitoreables.

  **Optimización de costos y latencia:** Incluir al menos una acción orientada a reducir el costo computacional o de inferencia (por ejemplo, selección dinámica de modelo, reducción de longitud de contexto,
  uso de caching o batching, o control de temperatura/token limit).

  **Acción final disparada automáticamente:** El sistema debe ejecutar una acción de salida automatizada, como enviar un email (vía Gmail API o SMTP), registrar eventos en un log externo, o integrar con
  servicios de terceros (Webhook, Slack, Notion, etc.).

  **Evaluación de desempeño y métricas:**

  - Performance del modelo: tiempo de respuesta, consumo de tokens, precisión semántica (BLEU/ROUGE/semantic similarity).
  - Performance de los agentes: coherencia de decisiones, calidad del diálogo entre agentes, redundancia evitada, latencia.
  - Eficiencia del RAG: recall, precisión@k, tiempo de recuperación, cobertura de contexto útil.

### Entregables

- Un informe en PDF que se llamara `informe_final.pdf` y se alojara en `llm-agent-project/docs/informe/informe_final.pdf`
- El codigo fuente (este repositorio)
- Una presentacion que se alojara en docs/presentacion
- Un video demo de la app que se llamara `app_demo.mp4` y se alojara en `llm-agent-project/docs/video/app_demo.mp4`
- El reporte de metricas en PDF se llama `metricas.pdf` y se aloja en `llm-agent-project/reports/metricas.pdf`


### Informe tecnico

Debe tener:

- Objetivo del proyecto
- Arquitectura general (diagrama de flujo + descripción de componentes)
- Implementación técnica (herramientas, módulos clave)
- Evaluación (métricas de desempeño de modelos, agentes y RAG)
- Resultados y ejemplos
- Conclusiones y mejoras futuras
- Planificación del equipo (tabla con tareas, responsables y estado)

Esto es informacion muy valiosa sobre nuestro proyecto que tambien tiene que estar en docs/architecture.html.
