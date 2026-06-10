from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from src.runtime.config import LLMModelSelection
from src.runtime.pipelines.factory import build_component_advisor
from tests.eval.test_metrics_evaluation import (
    THRESHOLDS,
    _abstention_correct,
    _catalog_by_sku,
    _context_recall_score,
    _load_json,
    _model_selection,
    _top_skus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "metrics_dataset.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "deepeval_metrics_report.html"


class MetadataBooleanMetric(BaseMetric):
    metric_key: str
    metric_name: str

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.score = None
        self.success = None
        self.reason = None
        self.async_mode = False

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        value = test_case.metadata.get(self.metric_key) if test_case.metadata else None
        self.score = 1.0 if value else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"{self.metric_name}: {'hit' if value else 'miss'}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return self.metric_name


class DeepEvalRecallAtKMetric(MetadataBooleanMetric):
    metric_key = "recall_hit"
    metric_name = "DeepEval Recall@K"


class DeepEvalCorrectAbstentionMetric(MetadataBooleanMetric):
    metric_key = "abstention_correct"
    metric_name = "DeepEval Correct Abstention"


class DeepEvalContextRecallMetric(BaseMetric):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.score = None
        self.success = None
        self.reason = None
        self.async_mode = False

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        self.score = float(test_case.metadata.get("context_score", 0.0) if test_case.metadata else 0.0)
        self.success = self.score >= self.threshold
        self.reason = "Fraction of required evidence fields present in the final answer."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "DeepEval Context Recall"


@dataclass(frozen=True)
class DeepEvalCaseResult:
    sample: dict[str, Any]
    test_case: LLMTestCase
    route: str
    top_skus: list[str]
    recall_score: float | None
    context_score: float | None
    abstention_score: float | None
    final_answer: str
    token_count: int
    ok: bool


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _require_real_llm() -> None:
    _load_env()
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required because eval_deepeval must call the real LLM pipeline.")


def _report_path() -> Path:
    configured = os.getenv("DEEPEVAL_REPORT_PATH")
    return Path(configured) if configured else DEFAULT_REPORT_PATH


def _make_test_case(sample: dict[str, Any], response: Any, catalog: dict[str, dict[str, Any]]) -> LLMTestCase:
    labels = response.metrics.get("labels", {})
    counters = response.metrics.get("counters", {})
    top_skus = _top_skus(response)
    expected_sku = sample.get("expected_sku")
    recall_hit = bool(expected_sku and expected_sku in top_skus) if "recall" in sample.get("metrics", []) else None
    context_score = (
        _context_recall_score(sample, response, catalog) if "context" in sample.get("metrics", []) else None
    )
    abstention_correct = _abstention_correct(response) if "abstention" in sample.get("metrics", []) else None
    retrieval_context = []
    if response.local:
        retrieval_context = [result.document for result in response.local.response.results]

    return LLMTestCase(
        name=sample["id"],
        input=sample["query"],
        actual_output=str(response.final_answer or ""),
        expected_output=str(sample.get("expected_sku") or sample.get("expected_route")),
        retrieval_context=retrieval_context,
        metadata={
            "sample": sample,
            "route": labels.get("route"),
            "top_skus": top_skus,
            "recall_hit": recall_hit,
            "context_score": context_score,
            "abstention_correct": abstention_correct,
            "tokens": int(counters.get("total_tokens", 0)),
        },
    )


def _evaluate_case(test_case: LLMTestCase) -> DeepEvalCaseResult:
    sample = test_case.metadata["sample"]
    metrics = sample.get("metrics", [])
    recall_score = None
    context_score = None
    abstention_score = None

    if "recall" in metrics:
        recall_score = DeepEvalRecallAtKMetric(THRESHOLDS["recall_at_k"]).measure(test_case)
    if "context" in metrics:
        context_score = DeepEvalContextRecallMetric(THRESHOLDS["context_recall"]).measure(test_case)
    if "abstention" in metrics:
        abstention_score = DeepEvalCorrectAbstentionMetric(THRESHOLDS["correct_abstention"]).measure(test_case)

    route_ok = test_case.metadata["route"] == sample.get("expected_route")
    recall_ok = recall_score is None or recall_score >= THRESHOLDS["recall_at_k"]
    context_ok = context_score is None or context_score > 0
    abstention_ok = abstention_score is None or abstention_score >= THRESHOLDS["correct_abstention"]
    return DeepEvalCaseResult(
        sample=sample,
        test_case=test_case,
        route=str(test_case.metadata["route"]),
        top_skus=list(test_case.metadata["top_skus"]),
        recall_score=recall_score,
        context_score=context_score,
        abstention_score=abstention_score,
        final_answer=str(test_case.actual_output or ""),
        token_count=int(test_case.metadata["tokens"]),
        ok=bool(route_ok and recall_ok and context_ok and abstention_ok),
    )


def _rate(values: list[float | None]) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return round(sum(usable) / len(usable), 4)


def _aggregate(results: list[DeepEvalCaseResult]) -> dict[str, float]:
    clean = [result.recall_score for result in results if result.sample.get("variant") == "clean"]
    variants = [result.recall_score for result in results if result.sample.get("variant") != "clean"]
    clean_recall = _rate(clean)
    variant_recall = _rate(variants)
    degradation = max(0.0, clean_recall - variant_recall)
    return {
        "recall_at_k": _rate([result.recall_score for result in results]),
        "context_recall": _rate([result.context_score for result in results]),
        "correct_abstention": _rate([result.abstention_score for result in results]),
        "robustness": round(1.0 - degradation, 4),
        "clean_recall": clean_recall,
        "variant_recall": variant_recall,
    }


def _write_html_report(results: list[DeepEvalCaseResult], summary: dict[str, float], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = sum(result.token_count for result in results)
    cards = "\n".join(_metric_card(name, summary[name], THRESHOLDS[name]) for name in THRESHOLDS)
    rows = "\n".join(_case_row(result) for result in results)
    report_path.write_text(
        f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte DeepEval - Component Purchase Advisor</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #172033; }}
    .muted {{ color: #667085; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; }}
    .pass {{ color: #067647; font-weight: 800; }}
    .fail {{ color: #b42318; font-weight: 800; }}
    .pass-bg {{ background: #ecfdf3; }}
    .fail-bg {{ background: #fef3f2; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e7ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f4f7; }}
    code {{ background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }}
    .answer {{ max-width: 420px; }}
  </style>
</head>
<body>
  <h1>Reporte DeepEval</h1>
  <p class="muted">Framework: <code>deepeval</code> con <code>LLMTestCase</code> y metricas custom <code>BaseMetric</code>. Dataset: <code>data/eval/metrics_dataset.json</code>. Muestras: {len(results)}. Tokens pipeline: {total_tokens}.</p>
  <div class="cards">{cards}</div>
  <p class="muted">Robustez = 1 - caida entre recall clean y recall con parafrasis/ruido. Clean recall: {summary["clean_recall"]:.3f}; variant recall: {summary["variant_recall"]:.3f}.</p>
  <table>
    <thead><tr><th>Estado</th><th>ID</th><th>Query</th><th>Esperado</th><th>Ruta</th><th>Top SKUs</th><th>Recall</th><th>Context</th><th>Abstencion</th><th>Tokens</th><th>Respuesta</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>""",
        encoding="utf-8",
    )


def _metric_card(metric: str, value: float, threshold: float) -> str:
    passed = value >= threshold
    status = "PASA" if passed else "FALLA"
    cls = "pass" if passed else "fail"
    bg = "pass-bg" if passed else "fail-bg"
    return (
        f'<div class="card {bg}"><h3>{html.escape(metric)}</h3>'
        f'<div class="{cls}">{status}</div>'
        f"<p>Score: {value:.3f}<br>Umbral: {threshold:.3f}</p></div>"
    )


def _case_row(result: DeepEvalCaseResult) -> str:
    status = "PASA" if result.ok else "FALLA"
    cls = "pass" if result.ok else "fail"
    return (
        "<tr>"
        f'<td class="{cls}">{status}</td>'
        f"<td>{html.escape(result.sample['id'])}</td>"
        f"<td>{html.escape(result.sample['query'])}</td>"
        f"<td>{html.escape(str(result.sample.get('expected_sku') or result.sample.get('expected_route')))}</td>"
        f"<td>{html.escape(result.route)}</td>"
        f"<td>{html.escape(', '.join(result.top_skus))}</td>"
        f"<td>{_fmt(result.recall_score)}</td>"
        f"<td>{_fmt(result.context_score)}</td>"
        f"<td>{_fmt(result.abstention_score)}</td>"
        f"<td>{result.token_count}</td>"
        f'<td class="answer">{html.escape(result.final_answer[:700])}</td>'
        "</tr>"
    )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


@pytest.mark.eval_deepeval
def test_deepeval_dataset_metrics_generate_html_report() -> None:
    _require_real_llm()
    samples = _load_json(DATASET_PATH)
    assert len(samples) >= 40, "Evaluation dataset must contain at least 40 samples."

    catalog = _catalog_by_sku()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()
    model_selection: LLMModelSelection = _model_selection()

    test_cases: list[LLMTestCase] = []
    for sample in samples:
        response = advisor.answer(
            sample["query"],
            top_k=int(os.getenv("EVAL_TOP_K", "3")),
            web_limit=int(os.getenv("EVAL_WEB_LIMIT", "1")),
            model_selection=model_selection,
        )
        test_cases.append(_make_test_case(sample, response, catalog))

    results = [_evaluate_case(test_case) for test_case in test_cases]
    summary = _aggregate(results)
    report_path = _report_path()
    _write_html_report(results, summary, report_path)

    failures = [
        f"{metric}={summary[metric]:.3f} < {threshold:.3f}"
        for metric, threshold in THRESHOLDS.items()
        if summary[metric] < threshold
    ]
    assert not failures, f"DeepEval metric evaluation failed; report written to {report_path}: {', '.join(failures)}"
