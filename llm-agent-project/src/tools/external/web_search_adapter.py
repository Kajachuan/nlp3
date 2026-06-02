from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "web_search_tool"


class SearchCallable(Protocol):
    def __call__(self, query: str, limit: int = 5) -> Any:
        ...


class WebSearchAdapter:
    """Adapter for https://github.com/lsaraco/web-search-tool/tree/main/web_search_tool.

    The real package is optional for this POC. When unavailable, the adapter returns
    deterministic demo results so the full agent pipeline remains executable.
    """

    def __init__(self, search_fn: SearchCallable | None = None) -> None:
        self.search_fn = search_fn or self._discover_search_fn()
        self.mode = "external" if self.search_fn else "offline-demo"

    def search(self, query: str, limit: int = 3) -> list[WebResult]:
        if self.search_fn:
            raw_results = self.search_fn(query=query, limit=limit)
            return self._normalize(raw_results)[:limit]
        return self._offline_results(query, limit)

    def _discover_search_fn(self) -> SearchCallable | None:
        candidates = [
            ("web_search_tool", "search"),
            ("web_search_tool.search", "search"),
            ("web_search_tool.web_search", "search"),
        ]
        for module_name, attr in candidates:
            try:
                module = __import__(module_name, fromlist=[attr])
                fn = getattr(module, attr, None)
                if callable(fn):
                    return fn
            except Exception:
                continue
        return None

    def _normalize(self, raw_results: Any) -> list[WebResult]:
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", [raw_results])
        normalized: list[WebResult] = []
        for result in raw_results or []:
            if isinstance(result, dict):
                normalized.append(
                    WebResult(
                        title=str(result.get("title") or result.get("name") or "Resultado web"),
                        url=str(result.get("url") or result.get("link") or ""),
                        snippet=str(result.get("snippet") or result.get("description") or result.get("content") or ""),
                    )
                )
        return normalized

    def _offline_results(self, query: str, limit: int) -> list[WebResult]:
        templates = [
            WebResult(
                title="Buenas practicas de desacople y alimentacion",
                url="https://example.com/electronics/decoupling",
                snippet="Usar capacitores ceramicos de 100 nF cerca de cada integrado reduce ruido y caidas transitorias.",
                source="offline-demo",
            ),
            WebResult(
                title="Seleccion de sensores ambientales I2C",
                url="https://example.com/electronics/i2c-sensors",
                snippet="Sensores como BME280 combinan temperatura, humedad y presion con bajo consumo y calibracion digital.",
                source="offline-demo",
            ),
            WebResult(
                title="ESP32 para prototipos IoT",
                url="https://example.com/electronics/esp32-iot",
                snippet="ESP32 integra WiFi, BLE y multiples GPIO, por lo que suele ser una opcion eficiente para telemetria.",
                source="offline-demo",
            ),
        ]
        terms = set(query.lower().split())
        ranked = sorted(
            templates,
            key=lambda item: len(terms.intersection(item.title.lower().split() + item.snippet.lower().split())),
            reverse=True,
        )
        return ranked[:limit]
