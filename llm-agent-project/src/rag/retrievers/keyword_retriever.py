from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.rag.retrievers.catalog_retriever import tokenize
from src.runtime.catalog_data import load_catalog_items
from src.vectorstore.chroma.catalog_store import ChromaCatalogStore


@dataclass(frozen=True)
class KeywordHit:
    item: dict[str, Any]
    score: float
    matched_terms: list[str]
    document: str


class KeywordCatalogRetriever:
    def __init__(self, catalog_source: Path | Sequence[dict[str, Any]]) -> None:
        self.catalog_path = catalog_source if isinstance(catalog_source, Path) else None
        self.items = load_catalog_items(catalog_source) if isinstance(catalog_source, Path) else list(catalog_source)
        self.documents = [ChromaCatalogStore.item_to_document_static(item) for item in self.items]
        self.doc_tokens = [tokenize(document) for document in self.documents]

    def search(self, query: str, top_k: int = 3) -> list[KeywordHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        hits = []
        for item, document, tokens in zip(self.items, self.documents, self.doc_tokens):
            matched_terms = sorted(query_terms.intersection(tokens))
            sku_bonus = self._sku_bonus(query, item)
            score = self._bm25_like_score(query_tokens, tokens) + sku_bonus
            if score <= 0 and not matched_terms:
                continue
            hits.append(
                KeywordHit(
                    item=item,
                    score=round(score, 4),
                    matched_terms=matched_terms,
                    document=document,
                )
            )

        hits.sort(key=lambda hit: (hit.score, hit.item.get("stock", 0)), reverse=True)
        return hits[:top_k]

    def _bm25_like_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        score = 0.0
        doc_len = len(doc_tokens)
        unique_docs = max(len(self.doc_tokens), 1)
        for token in query_tokens:
            term_frequency = doc_tokens.count(token)
            if term_frequency == 0:
                continue
            docs_with_term = sum(1 for tokens in self.doc_tokens if token in tokens)
            idf = 1.0 + (unique_docs / max(docs_with_term, 1))
            score += (term_frequency / doc_len) * idf
        coverage = len(set(query_tokens).intersection(doc_tokens)) / max(len(set(query_tokens)), 1)
        return score + (0.25 * coverage)

    def _sku_bonus(self, query: str, item: dict[str, Any]) -> float:
        sku = str(item.get("sku", "")).lower()
        name = str(item.get("name", "")).lower()
        lowered = query.lower()
        bonus = 0.0
        if sku and sku in lowered:
            bonus += 1.0
        if name and name in lowered:
            bonus += 0.7
        return bonus
