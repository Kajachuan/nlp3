from pathlib import Path
from uuid import uuid4

from src.rag.retrievers.catalog_retriever import CatalogRetriever
from src.vectorstore.chroma.catalog_store import ChromaCatalogStore


def test_catalog_retriever_finds_sensor() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog_path = root / "data" / "raw" / "electronic_components_catalog.json"
    store = ChromaCatalogStore(root / "tmp" / f"test-chroma-{uuid4().hex}")
    store.ensure_catalog_indexed(catalog_path)
    retriever = CatalogRetriever(store)

    hits = retriever.search("sensor temperatura humedad i2c", top_k=2)

    assert hits
    assert hits[0].item["sku"] == "S-BME280-I2C"
    assert hits[0].score > 0
