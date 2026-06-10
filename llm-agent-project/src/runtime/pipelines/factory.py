from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src.agents.cores.local_hybrid_agent import LocalHybridSearchAgent
from src.agents.cores.web_research_agent import WebResearchAgent
from src.agents.orchestration.evidence_gate import LocalEvidenceGate
from src.agents.orchestration.component_advisor import ComponentAdvisorOrchestrator
from src.rag.retrievers.catalog_retriever import CatalogRetriever
from src.rag.retrievers.hybrid_retriever import HybridCatalogRetriever
from src.rag.retrievers.keyword_retriever import KeywordCatalogRetriever
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
    semantic_retriever = CatalogRetriever(catalog_store)
    keyword_retriever = KeywordCatalogRetriever(catalog_path)
    hybrid_retriever = HybridCatalogRetriever(semantic_retriever, keyword_retriever)
    return ComponentAdvisorOrchestrator(
        local_hybrid_agent=LocalHybridSearchAgent(hybrid_retriever),
        web_agent=WebResearchAgent(WebSearchAdapter()),
        evidence_gate=LocalEvidenceGate(),
        audit_log_path=audit_log_path,
    )
