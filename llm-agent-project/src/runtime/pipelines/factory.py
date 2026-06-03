from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.agents.cores.catalog_agent import CatalogAgent
from src.agents.cores.web_research_agent import WebResearchAgent
from src.agents.orchestration.component_advisor import ComponentAdvisorOrchestrator
from src.rag.retrievers.catalog_retriever import CatalogRetriever
from src.runtime.providers.pretrained_intent_model import PretrainedIntentModel
from src.tools.external.web_search_adapter import WebSearchAdapter
from src.vectorstore.chroma.catalog_store import ChromaCatalogStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def build_component_advisor() -> ComponentAdvisorOrchestrator:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    catalog_path = PROJECT_ROOT / "data" / "raw" / "electronic_components_catalog.json"
    chroma_path = PROJECT_ROOT / "data" / "processed" / "chroma"
    audit_log_path = PROJECT_ROOT / "logs" / "advisor_audit.jsonl"
    catalog_store = ChromaCatalogStore(chroma_path)
    catalog_store.ensure_catalog_indexed(catalog_path)
    retriever = CatalogRetriever(catalog_store)
    return ComponentAdvisorOrchestrator(
        catalog_agent=CatalogAgent(retriever),
        web_agent=WebResearchAgent(WebSearchAdapter()),
        intent_model=PretrainedIntentModel(),
        audit_log_path=audit_log_path,
    )
