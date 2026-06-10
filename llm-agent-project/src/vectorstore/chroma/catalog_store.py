from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from src.runtime.catalog_data import load_catalog_items
from src.vectorstore.chroma.hash_embedding import HashEmbeddingFunction


class ChromaCatalogStore:
    def __init__(
        self,
        persist_path: Path,
        collection_name: str = "electronic_components_catalog",
    ) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_function = HashEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_catalog_indexed(self, catalog_path: Path) -> None:
        items = load_catalog_items(catalog_path)
        self.ensure_items_indexed(items)

    def ensure_items_indexed(self, items: list[dict[str, Any]]) -> None:
        existing = self.collection.count()
        if existing == len(items):
            return

        if existing:
            current = self.collection.get(include=[])
            self.collection.delete(ids=current["ids"])
        ids = [item["sku"] for item in items]
        documents = [self.item_to_document(item) for item in items]
        metadatas = [self.item_to_metadata(item) for item in items]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        rows = []
        for row_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            rows.append(
                {
                    "id": row_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": float(distance),
                    "item": json.loads(metadata["item_json"]),
                }
            )
        return rows

    def item_to_document(self, item: dict[str, Any]) -> str:
        return self.item_to_document_static(item)

    @staticmethod
    def item_to_document_static(item: dict[str, Any]) -> str:
        specs = " ".join(f"{key}: {value}" for key, value in item.get("specs", {}).items())
        return " ".join(
            [
                str(item.get("sku", "")),
                str(item.get("name", "")),
                f"categoria: {item.get('category', '')}",
                str(item.get("description", "")),
                specs,
                str(item.get("recommended_use", "")),
            ]
        )

    def item_to_metadata(self, item: dict[str, Any]) -> dict[str, str | int | float | bool]:
        return {
            "sku": str(item.get("sku", "")),
            "name": str(item.get("name", "")),
            "category": str(item.get("category", "")),
            "stock": int(item.get("stock", 0)),
            "price_usd": float(item.get("price_usd", 0.0)),
            "item_json": json.dumps(item, ensure_ascii=True),
        }

