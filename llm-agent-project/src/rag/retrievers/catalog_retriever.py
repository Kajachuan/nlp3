from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a",
    "al",
    "and",
    "con",
    "de",
    "del",
    "el",
    "en",
    "for",
    "la",
    "las",
    "lo",
    "los",
    "medir",
    "necesito",
    "para",
    "por",
    "que",
    "the",
    "un",
    "una",
    "uso",
    "y",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


@dataclass(frozen=True)
class CatalogHit:
    item: dict[str, Any]
    score: float
    matched_terms: list[str]


class CatalogRetriever:
    """Small lexical retriever with reranking for the POC catalog."""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        self.items = self._load_catalog(catalog_path)
        self.documents = [self._item_to_document(item) for item in self.items]
        self.doc_tokens = [tokenize(document) for document in self.documents]
        self.idf = self._build_idf(self.doc_tokens)

    def search(self, query: str, top_k: int = 3) -> list[CatalogHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        hits: list[CatalogHit] = []
        for item, tokens in zip(self.items, self.doc_tokens):
            score = self._score(query_tokens, tokens)
            matched_terms = sorted(query_terms.intersection(tokens))
            if score > 0:
                hits.append(CatalogHit(item=item, score=round(score, 4), matched_terms=matched_terms))

        hits.sort(key=lambda hit: (hit.score, hit.item.get("stock", 0)), reverse=True)
        return hits[:top_k]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        doc_len = max(len(doc_tokens), 1)
        doc_counts = {token: doc_tokens.count(token) for token in set(doc_tokens)}
        score = 0.0
        for token in query_tokens:
            tf = doc_counts.get(token, 0) / doc_len
            score += tf * self.idf.get(token, 1.0)

        query_terms = set(query_tokens)
        coverage_bonus = 0.12 * (len(query_terms.intersection(doc_tokens)) / max(len(query_terms), 1))
        phrase_bonus = self._phrase_bonus(query_tokens, doc_tokens)
        return score + coverage_bonus + phrase_bonus

    def _phrase_bonus(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if len(query_tokens) < 2:
            return 0.0
        query_pairs = set(zip(query_tokens, query_tokens[1:]))
        doc_pairs = set(zip(doc_tokens, doc_tokens[1:]))
        return 0.04 * len(query_pairs.intersection(doc_pairs))

    def _build_idf(self, docs: list[list[str]]) -> dict[str, float]:
        total_docs = len(docs)
        vocabulary = {token for doc in docs for token in set(doc)}
        idf: dict[str, float] = {}
        for token in vocabulary:
            containing = sum(1 for doc in docs if token in doc)
            idf[token] = math.log((1 + total_docs) / (1 + containing)) + 1
        return idf

    def _load_catalog(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("El catalogo debe ser una lista JSON.")
        return data

    def _item_to_document(self, item: dict[str, Any]) -> str:
        specs = " ".join(f"{key} {value}" for key, value in item.get("specs", {}).items())
        return " ".join(
            [
                str(item.get("sku", "")),
                str(item.get("name", "")),
                str(item.get("category", "")),
                str(item.get("description", "")),
                specs,
                str(item.get("recommended_use", "")),
            ]
        )
