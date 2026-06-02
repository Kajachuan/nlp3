from pathlib import Path

from src.rag.retrievers.catalog_retriever import CatalogRetriever


def test_catalog_retriever_finds_sensor() -> None:
    root = Path(__file__).resolve().parents[2]
    retriever = CatalogRetriever(root / "data" / "raw" / "electronic_components_catalog.json")

    hits = retriever.search("sensor temperatura humedad i2c", top_k=2)

    assert hits
    assert hits[0].item["sku"] == "S-BME280-I2C"
    assert hits[0].score > 0
