from __future__ import annotations

import hashlib
import math
import re


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class HashEmbeddingFunction:
    """Small local embedding function for Chroma without model downloads.

    It is intentionally simple: token hashes are projected into a fixed-size vector
    and L2-normalized. Chroma remains the real vector index; the embedding backend
    can be swapped later for SentenceTransformers or an API model.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        return "hash-embedding"

    def is_legacy(self) -> bool:
        return False

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, int]:
        return {"dimensions": self.dimensions}

    @classmethod
    def build_from_config(cls, config: dict[str, int]) -> "HashEmbeddingFunction":
        return cls(dimensions=int(config.get("dimensions", 384)))

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
