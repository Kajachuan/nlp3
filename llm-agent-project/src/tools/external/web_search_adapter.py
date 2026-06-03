from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "web_search_tool"
    price: str | None = None
    vendor: str | None = None
    score: float | None = None
    raw: dict[str, Any] | None = None


WebSearchMethod = Literal["google_shopping", "google_search", "tavily"]


class WebSearchAdapter:
    """Adapter for https://github.com/lsaraco/web-search-tool/tree/main/web_search_tool.

    Uses the package's WebSearchClient.search(...) API. The offline fallback only
    runs when the package is missing or required backend credentials are absent.
    """

    def __init__(
        self,
        method: WebSearchMethod | None = None,
        preferred_sites: list[str] | None = None,
        preferred_only: bool = False,
        include_delivery_details: bool = False,
    ) -> None:
        self._disable_dead_local_proxy()
        self.method = method or os.getenv("WEB_SEARCH_DEFAULT_METHOD", "tavily")
        self.preferred_sites = preferred_sites or self._preferred_sites_from_env()
        self.preferred_only = preferred_only
        self.include_delivery_details = include_delivery_details
        self.client = self._build_client()
        self.mode = self._resolve_mode()
        self.last_mode = self.mode

    def search(self, query: str, limit: int = 3) -> list[WebResult]:
        return self.search_with_options(query=query, limit=limit)

    def search_with_options(
        self,
        query: str,
        limit: int = 3,
        method: WebSearchMethod | None = None,
        preferred_sites: list[str] | None = None,
        preferred_only: bool | None = None,
        include_delivery_details: bool | None = None,
    ) -> list[WebResult]:
        method = method or self.method
        preferred_sites = preferred_sites if preferred_sites is not None else self.preferred_sites
        preferred_only = self.preferred_only if preferred_only is None else preferred_only
        include_delivery_details = (
            self.include_delivery_details if include_delivery_details is None else include_delivery_details
        )
        mode = self._resolve_mode(method)
        self.last_mode = mode

        if self.client and mode == "web_search_tool":
            try:
                raw_results = self.client.search(
                    query=query,
                    method=method,
                    limit=limit,
                    preferred_sites=preferred_sites,
                    preferred_only=preferred_only,
                    include_delivery_details=include_delivery_details,
                )
                normalized = self._normalize(raw_results, method=method)
                if self._is_credential_error(normalized):
                    return self._offline_results(query, limit, reason=normalized[0].snippet)
                return normalized[:limit]
            except Exception as exc:
                self.last_mode = "offline-demo: tool error"
                return self._offline_results(query, limit, reason=self._safe_error(exc))
        return self._offline_results(query, limit, reason=mode)

    def _build_client(self):
        try:
            from web_search_tool import WebSearchClient

            return WebSearchClient()
        except Exception:
            return None

    def _resolve_mode(self, method: WebSearchMethod | None = None) -> str:
        method = method or self.method
        if not self.client:
            return "offline-demo: package not installed"
        if method == "tavily" and not os.getenv("TAVILY_API_KEY"):
            return "offline-demo: missing TAVILY_API_KEY"
        if method == "google_shopping" and not os.getenv("SERPAPI_API_KEY"):
            return "offline-demo: missing SERPAPI_API_KEY"
        if method == "google_search":
            missing = [key for key in ["SERPAPI_API_KEY", "ZYTE_API_KEY"] if not os.getenv(key)]
            if missing:
                return f"offline-demo: missing {', '.join(missing)}"
        return "web_search_tool"

    def _normalize(self, raw_results: Any, method: str | None = None) -> list[WebResult]:
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results", [raw_results])
        normalized: list[WebResult] = []
        for result in raw_results or []:
            if isinstance(result, dict):
                url = result.get("url") or result.get("link") or result.get("source_url") or result.get("product_link") or ""
                snippet = (
                    result.get("snippet")
                    or result.get("description")
                    or result.get("content")
                    or result.get("delivery_summary")
                    or result.get("error")
                    or ""
                )
                normalized.append(
                    WebResult(
                        title=str(result.get("title") or result.get("name") or result.get("product_title") or "Resultado web"),
                        url=str(url),
                        snippet=str(snippet),
                        source=str(result.get("source") or result.get("vendor") or method or self.method),
                        price=self._string_or_none(result.get("price") or result.get("extracted_price")),
                        vendor=self._string_or_none(result.get("vendor") or result.get("source")),
                        score=self._float_or_none(result.get("score") or result.get("rating")),
                        raw=result,
                    )
                )
        return normalized

    def _offline_results(self, query: str, limit: int, reason: str | None = None) -> list[WebResult]:
        templates = [
            WebResult(
                title="Modo demo: configure credenciales para busqueda real",
                url="https://github.com/lsaraco/web-search-tool/tree/main/web_search_tool",
                snippet=reason or self.mode,
                source="offline-demo",
            ),
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
            templates[1:],
            key=lambda item: len(terms.intersection(item.title.lower().split() + item.snippet.lower().split())),
            reverse=True,
        )
        return [templates[0], *ranked][:limit]

    def _preferred_sites_from_env(self) -> list[str] | None:
        sites = os.getenv("WEB_SEARCH_SITES", "")
        parsed = [site.strip() for site in sites.split(",") if site.strip()]
        return parsed or None

    def _is_credential_error(self, results: list[WebResult]) -> bool:
        if not results:
            return False
        first = results[0]
        return first.raw is not None and first.raw.get("status") == "error"

    def _string_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _float_or_none(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_error(self, exc: Exception) -> str:
        message = f"{type(exc).__name__}: {exc}"
        message = re.sub(r"(api_key=)[^&\\s()]+", r"\1[REDACTED]", message)
        message = re.sub(r"(SERPAPI_API_KEY=)[^&\\s()]+", r"\1[REDACTED]", message)
        return message

    def _disable_dead_local_proxy(self) -> None:
        if os.getenv("WEB_SEARCH_USE_SYSTEM_PROXY", "").lower() in {"1", "true", "yes"}:
            return

        dead_proxy_values = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
        for name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            if os.getenv(name) in dead_proxy_values:
                os.environ.pop(name, None)
