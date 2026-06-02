from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.vectorstore.chroma.catalog_store import ChromaCatalogStore


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
    """Catalog retriever backed by a persistent Chroma collection."""

    def __init__(self, store: ChromaCatalogStore) -> None:
        self.store = store

    def search(self, query: str, top_k: int = 3) -> list[CatalogHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        hits: list[CatalogHit] = []
        rows = self.store.query(query, top_k=top_k)
        for row in rows:
            item = row["item"]
            tokens = tokenize(row["document"])
            vector_score = max(0.0, 1.0 - row["distance"])
            lexical_score = self._score(query_tokens, tokens)
            score = (0.75 * vector_score) + (0.25 * lexical_score)
            matched_terms = sorted(query_terms.intersection(tokens))
            hits.append(CatalogHit(item=item, score=round(score, 4), matched_terms=matched_terms))

        hits.sort(key=lambda hit: (hit.score, hit.item.get("stock", 0)), reverse=True)
        return hits[:top_k]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        doc_len = max(len(doc_tokens), 1)
        doc_counts = {token: doc_tokens.count(token) for token in set(doc_tokens)}
        score = 0.0
        for token in query_tokens:
            score += doc_counts.get(token, 0) / doc_len

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
