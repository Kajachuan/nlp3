from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.rag.retrievers.catalog_retriever import CatalogHit, CatalogRetriever
from src.rag.retrievers.keyword_retriever import KeywordCatalogRetriever, KeywordHit


@dataclass(frozen=True)
class HybridSearchResult:
    item: dict[str, Any]
    score: float
    semantic_score: float
    keyword_score: float
    matched_terms: list[str]
    sources: list[str]
    document: str


@dataclass(frozen=True)
class HybridSearchResponse:
    results: list[HybridSearchResult]
    rag_hits: list[CatalogHit]
    keyword_hits: list[KeywordHit]
    timings_ms: dict[str, float]


class HybridCatalogRetriever:
    def __init__(self, semantic: CatalogRetriever, keyword: KeywordCatalogRetriever) -> None:
        self.semantic = semantic
        self.keyword = keyword

    def search(self, query: str, top_k: int = 3) -> HybridSearchResponse:
        timings_ms: dict[str, float] = {}
        started = time.perf_counter()
        rag_hits = self.semantic.search(query, top_k=top_k)
        timings_ms["rag_search"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        keyword_hits = self.keyword.search(query, top_k=top_k)
        timings_ms["keyword_bm25_search"] = round((time.perf_counter() - started) * 1000, 2)

        started = time.perf_counter()
        merged: dict[str, dict[str, Any]] = {}

        for hit in rag_hits:
            sku = str(hit.item.get("sku", ""))
            merged[sku] = {
                "item": hit.item,
                "semantic_score": hit.score,
                "keyword_score": 0.0,
                "matched_terms": set(hit.matched_terms),
                "sources": {"rag"},
                "document": self.semantic.store.item_to_document(hit.item),
            }

        for hit in keyword_hits:
            sku = str(hit.item.get("sku", ""))
            row = merged.setdefault(
                sku,
                {
                    "item": hit.item,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "matched_terms": set(),
                    "sources": set(),
                    "document": hit.document,
                },
            )
            row["keyword_score"] = max(row["keyword_score"], hit.score)
            row["matched_terms"].update(hit.matched_terms)
            row["sources"].add("keyword")

        results = []
        for row in merged.values():
            semantic_score = float(row["semantic_score"])
            keyword_score = float(row["keyword_score"])
            score = (0.65 * semantic_score) + (0.35 * keyword_score)
            if {"rag", "keyword"}.issubset(row["sources"]):
                score += 0.08
            results.append(
                HybridSearchResult(
                    item=row["item"],
                    score=round(score, 4),
                    semantic_score=round(semantic_score, 4),
                    keyword_score=round(keyword_score, 4),
                    matched_terms=sorted(row["matched_terms"]),
                    sources=sorted(row["sources"]),
                    document=str(row["document"]),
                )
            )

        results.sort(key=lambda result: (result.score, result.item.get("stock", 0)), reverse=True)
        timings_ms["hybrid_merge"] = round((time.perf_counter() - started) * 1000, 2)
        return HybridSearchResponse(
            results=results[:top_k],
            rag_hits=rag_hits,
            keyword_hits=keyword_hits,
            timings_ms=timings_ms,
        )
