from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


class LangfuseTracer:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self.client = None
        self.last_error: str | None = None
        if self.enabled:
            try:
                from langfuse import Langfuse

                host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
                self.client = Langfuse(host=host)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.enabled = False

    @contextmanager
    def trace(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator["TraceContext"]:
        context = TraceContext(self.client, None, self)
        if self.client:
            try:
                context.trace = self.client.trace(name=name, metadata=metadata or {})
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                context.trace = None
        try:
            yield context
        finally:
            if self.client:
                try:
                    self.client.flush()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    pass


class TraceContext:
    def __init__(self, client: Any, trace: Any, owner: LangfuseTracer) -> None:
        self.client = client
        self.trace = trace
        self.owner = owner

    @property
    def trace_id(self) -> str | None:
        if not self.trace:
            return None
        value = getattr(self.trace, "id", None) or getattr(self.trace, "trace_id", None)
        return str(value) if value else None

    @property
    def trace_url(self) -> str | None:
        if not self.trace:
            return None
        try:
            value = self.trace.get_trace_url()
        except Exception as exc:
            self.owner.last_error = f"{type(exc).__name__}: {exc}"
            return None
        return str(value) if value else None

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        span = None
        if self.trace:
            try:
                span = self.trace.span(name=name, metadata=metadata or {})
            except Exception as exc:
                self.owner.last_error = f"{type(exc).__name__}: {exc}"
                span = None
        try:
            yield
        finally:
            if span:
                try:
                    span.end()
                except Exception as exc:
                    self.owner.last_error = f"{type(exc).__name__}: {exc}"
                    pass

    def generation(
        self,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.trace:
            return
        try:
            self.trace.generation(
                name=name,
                model=model,
                input=input_text,
                output=output_text,
                metadata=metadata or {},
            )
        except Exception as exc:
            self.owner.last_error = f"{type(exc).__name__}: {exc}"
            pass
