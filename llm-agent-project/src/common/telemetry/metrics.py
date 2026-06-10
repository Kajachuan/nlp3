from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Metrics:
    timings_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str | bool | int | float] = field(default_factory=dict)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000, 2)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def set_score(self, name: str, value: float) -> None:
        self.scores[name] = round(float(value), 4)

    def set_label(self, name: str, value: str | bool | int | float) -> None:
        self.labels[name] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "timings_ms": self.timings_ms,
            "counters": self.counters,
            "scores": self.scores,
            "labels": self.labels,
        }


def append_audit_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": round(time.time(), 3), **event}
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True) + "\n")
