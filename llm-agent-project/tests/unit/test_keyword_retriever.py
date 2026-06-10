from pathlib import Path

from src.rag.retrievers.keyword_retriever import KeywordCatalogRetriever


def test_keyword_retriever_finds_by_sku() -> None:
    root = Path(__file__).resolve().parents[2]
    retriever = KeywordCatalogRetriever(root / "data" / "raw" / "electronic_components_catalog.json")

    hits = retriever.search("REG-AMS1117-3V3", top_k=2)

    assert hits
    assert hits[0].item["sku"] == "REG-AMS1117-3V3"
    assert hits[0].score > 0
