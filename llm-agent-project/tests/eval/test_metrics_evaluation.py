from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from src.runtime.config import LLMModelSelection
from src.runtime.pipelines.factory import build_component_advisor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "metrics_dataset.json"
CATALOG_PATH = PROJECT_ROOT / "data" / "raw" / "electronic_components_catalog.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "eval_metrics_report.html"

THRESHOLDS = {
    "recall_at_k": 0.85,
    "context_recall": 0.85,
    "correct_abstention": 0.90,
    "robustness": 0.90,
}


@dataclass(frozen=True)
class CaseResult:
    sample: dict[str, Any]
    ok: bool
    route: str
    final_answer: str
    recall_hit: bool | None
    context_score: float | None
    abstention_correct: bool | None
    top_skus: list[str]
    metrics: dict[str, Any]
    error: str | None = None


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def _require_real_llm() -> None:
    _load_env()
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required because eval_metrics must call the real LLM pipeline.")


def _model_selection() -> LLMModelSelection:
    model = os.getenv("EVAL_GROQ_MODEL") or os.getenv("E2E_GROQ_MODEL") or os.getenv("PLANNER_MODEL")
    model = model or "llama-3.3-70b-versatile"
    return LLMModelSelection(
        planner_model=os.getenv("EVAL_PLANNER_MODEL") or model,
        direct_response_model=os.getenv("EVAL_DIRECT_RESPONSE_MODEL") or os.getenv("DIRECT_RESPONSE_MODEL") or model,
        verifier_model=os.getenv("EVAL_VERIFIER_MODEL") or os.getenv("VERIFIER_MODEL") or model,
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _catalog_by_sku() -> dict[str, dict[str, Any]]:
    return {str(item["sku"]): item for item in _load_json(CATALOG_PATH)}


def _contains_number(text: str, value: Any) -> bool:
    if value is None:
        return False
    normalized_text = text.lower().replace(",", ".")
    raw = str(value).lower()
    candidates = {raw}
    try:
        numeric = float(value)
        candidates.add(f"{numeric:g}".lower())
        candidates.add(f"{numeric:.2f}".rstrip("0").rstrip(".").lower())
        candidates.add(f"{numeric:.3f}".rstrip("0").rstrip(".").lower())
    except (TypeError, ValueError):
        pass
    return any(candidate and candidate in normalized_text for candidate in candidates)


def _context_recall_score(sample: dict[str, Any], response: Any, catalog: dict[str, dict[str, Any]]) -> float:
    expected_sku = sample.get("expected_sku")
    if not expected_sku:
        return 0.0
    final = str(response.final_answer or "")
    lower_final = final.lower()
    aggregator = response.aggregator_decision

    citations = " ".join(getattr(aggregator, "citations", []) or []) if aggregator else ""
    source = str(getattr(aggregator, "source", "") or "") if aggregator else ""
    vendor = str(getattr(aggregator, "vendor", "") or "") if aggregator else ""
    price = str(getattr(aggregator, "price", "") or "") if aggregator else ""
    purchase_url = str(getattr(aggregator, "purchase_url", "") or "") if aggregator else ""
    accepted_items = _accepted_catalog_items(sample, catalog, _top_skus(response))

    checks = [
        any(
            str(item.get("sku", "")).lower() in lower_final
            or str(item.get("sku", "")).lower() in citations.lower()
            or str(item.get("name", "")).lower() in lower_final
            for item in accepted_items
        ),
        any(
            _contains_number(final, item.get("price_usd")) or _contains_number(price, item.get("price_usd"))
            for item in accepted_items
        ),
        "http://" in lower_final
        or "https://" in lower_final
        or "url" in lower_final
        or "link" in lower_final
        or "componentes.html" in purchase_url.lower(),
        bool(citations.strip()) or "local" in source.lower() or "catalog" in vendor.lower() or "catalog" in source.lower(),
    ]
    return round(sum(1 for check in checks if check) / len(checks), 4)


def _accepted_skus(sample: dict[str, Any], top_skus: list[str] | None = None) -> set[str]:
    accepted = {str(sample.get("expected_sku", ""))}
    accepted.update(str(sku) for sku in sample.get("acceptable_skus", []) if str(sku).strip())
    prefixes = [str(prefix) for prefix in sample.get("acceptable_sku_prefixes", []) if str(prefix).strip()]
    for sku in top_skus or []:
        if any(sku.startswith(prefix) for prefix in prefixes):
            accepted.add(sku)
    return {sku for sku in accepted if sku}


def _accepted_catalog_items(
    sample: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    top_skus: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [catalog[sku] for sku in _accepted_skus(sample, top_skus) if sku in catalog]


def _abstention_correct(response: Any) -> bool:
    labels = response.metrics.get("labels", {})
    counters = response.metrics.get("counters", {})
    return (
        labels.get("route") == "out_of_domain"
        and not counters.get("rag_result_count")
        and not counters.get("keyword_result_count")
        and not counters.get("web_result_count")
        and bool(str(response.final_answer or "").strip())
    )


def _top_skus(response: Any) -> list[str]:
    if not response.local:
        return []
    return [str(result.item.get("sku", "")) for result in response.local.response.results]


def _run_case(advisor: Any, sample: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> CaseResult:
    response = advisor.answer(
        sample["query"],
        top_k=int(os.getenv("EVAL_TOP_K", "3")),
        web_limit=int(os.getenv("EVAL_WEB_LIMIT", "1")),
        model_selection=_model_selection(),
    )
    labels = response.metrics.get("labels", {})
    top_skus = _top_skus(response)
    expected_sku = sample.get("expected_sku")
    recall_hit = None
    context_score = None
    abstention_correct = None

    if "recall" in sample.get("metrics", []):
        recall_hit = bool(expected_sku and _accepted_skus(sample, top_skus).intersection(top_skus))
    if "context" in sample.get("metrics", []):
        context_score = _context_recall_score(sample, response, catalog)
    if "abstention" in sample.get("metrics", []):
        abstention_correct = _abstention_correct(response)

    route_ok = labels.get("route") == sample.get("expected_route")
    local_ok = recall_hit is not False if recall_hit is not None else True
    abstention_ok = abstention_correct is not False if abstention_correct is not None else True
    return CaseResult(
        sample=sample,
        ok=bool(response.ok and route_ok and local_ok and abstention_ok),
        route=str(labels.get("route", "")),
        final_answer=str(response.final_answer or ""),
        recall_hit=recall_hit,
        context_score=context_score,
        abstention_correct=abstention_correct,
        top_skus=top_skus,
        metrics=response.metrics,
    )


def _aggregate(results: list[CaseResult]) -> dict[str, float]:
    recall_cases = [result for result in results if result.recall_hit is not None]
    context_cases = [result for result in results if result.context_score is not None]
    abstention_cases = [result for result in results if result.abstention_correct is not None]

    clean_recall = [
        result.recall_hit
        for result in recall_cases
        if result.sample.get("variant") == "clean" and result.recall_hit is not None
    ]
    non_clean_recall = [
        result.recall_hit
        for result in recall_cases
        if result.sample.get("variant") != "clean" and result.recall_hit is not None
    ]
    clean_rate = _rate(clean_recall)
    non_clean_rate = _rate(non_clean_recall)
    degradation = max(0.0, clean_rate - non_clean_rate)

    return {
        "recall_at_k": _rate([result.recall_hit for result in recall_cases]),
        "context_recall": _average([result.context_score for result in context_cases]),
        "correct_abstention": _rate([result.abstention_correct for result in abstention_cases]),
        "robustness": round(1.0 - degradation, 4),
        "clean_recall": clean_rate,
        "variant_recall": non_clean_rate,
    }


def _rate(values: list[bool | None]) -> float:
    usable = [value for value in values if value is not None]
    if not usable:
        return 0.0
    return round(sum(1 for value in usable if value) / len(usable), 4)


def _average(values: list[float | None]) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return round(sum(usable) / len(usable), 4)


def _pass(metric: str, value: float) -> bool:
    return value >= THRESHOLDS[metric]


def _report_path() -> Path:
    configured = os.getenv("EVAL_REPORT_PATH")
    return Path(configured) if configured else DEFAULT_REPORT_PATH


def _write_html_report(results: list[CaseResult], summary: dict[str, float], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(_case_row(result) for result in results)
    metric_cards = "\n".join(
        _metric_card(metric, summary[metric], THRESHOLDS[metric]) for metric in THRESHOLDS
    )
    total_tokens = sum(int(result.metrics.get("counters", {}).get("total_tokens", 0)) for result in results)
    total_cost = sum(float(result.metrics.get("labels", {}).get("estimated_cost_usd", 0.0) or 0.0) for result in results)
    html_doc = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte de metricas - Component Purchase Advisor</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #172033; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #667085; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; }}
    .pass {{ color: #067647; font-weight: 800; }}
    .fail {{ color: #b42318; font-weight: 800; }}
    .card.pass-bg {{ background: #ecfdf3; }}
    .card.fail-bg {{ background: #fef3f2; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e7ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f4f7; }}
    code {{ background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }}
    .answer {{ max-width: 420px; white-space: normal; }}
  </style>
</head>
<body>
  <h1>Reporte de metricas</h1>
  <p class="muted">Dataset: <code>{html.escape(str(DATASET_PATH.relative_to(PROJECT_ROOT)))}</code> · muestras: {len(results)} · tokens: {total_tokens} · costo estimado: USD {total_cost:.6f}</p>
  <div class="cards">{metric_cards}</div>
  <p class="muted">Robustez = 1 - caida entre recall clean y recall con parafrasis/ruido. Clean recall: {summary["clean_recall"]:.3f}; variant recall: {summary["variant_recall"]:.3f}.</p>
  <table>
    <thead>
      <tr>
        <th>Estado</th><th>ID</th><th>Query</th><th>Esperado</th><th>Ruta</th><th>Top SKUs</th>
        <th>Recall</th><th>Context</th><th>Abstencion</th><th>Tokens</th><th>Respuesta</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
    report_path.write_text(html_doc, encoding="utf-8")


def _metric_card(metric: str, value: float, threshold: float) -> str:
    passed = _pass(metric, value)
    status = "PASA" if passed else "FALLA"
    cls = "pass" if passed else "fail"
    bg = "pass-bg" if passed else "fail-bg"
    return (
        f'<div class="card {bg}"><h3>{html.escape(metric)}</h3>'
        f'<div class="{cls}">{status}</div>'
        f"<p>Score: {value:.3f}<br>Umbral: {threshold:.3f}</p></div>"
    )


def _case_row(result: CaseResult) -> str:
    status = "PASA" if result.ok else "FALLA"
    cls = "pass" if result.ok else "fail"
    counters = result.metrics.get("counters", {})
    return (
        "<tr>"
        f'<td class="{cls}">{status}</td>'
        f"<td>{html.escape(result.sample['id'])}</td>"
        f"<td>{html.escape(result.sample['query'])}</td>"
        f"<td>{html.escape(str(result.sample.get('expected_sku') or result.sample.get('expected_route')))}</td>"
        f"<td>{html.escape(result.route)}</td>"
        f"<td>{html.escape(', '.join(result.top_skus))}</td>"
        f"<td>{_format_optional_bool(result.recall_hit)}</td>"
        f"<td>{'' if result.context_score is None else f'{result.context_score:.3f}'}</td>"
        f"<td>{_format_optional_bool(result.abstention_correct)}</td>"
        f"<td>{int(counters.get('total_tokens', 0))}</td>"
        f'<td class="answer">{html.escape(result.final_answer[:700])}</td>'
        "</tr>"
    )


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


@pytest.mark.eval_metrics
def test_dataset_metrics_generate_html_report() -> None:
    _require_real_llm()
    samples = _load_json(DATASET_PATH)
    assert len(samples) >= 40, "Evaluation dataset must contain at least 40 samples."

    catalog = _catalog_by_sku()
    build_component_advisor.cache_clear()
    advisor = build_component_advisor()

    results: list[CaseResult] = []
    for sample in samples:
        try:
            results.append(_run_case(advisor, sample, catalog))
        except Exception as exc:  # Report failures per sample instead of losing the whole report.
            results.append(
                CaseResult(
                    sample=sample,
                    ok=False,
                    route="error",
                    final_answer="",
                    recall_hit=False if "recall" in sample.get("metrics", []) else None,
                    context_score=0.0 if "context" in sample.get("metrics", []) else None,
                    abstention_correct=False if "abstention" in sample.get("metrics", []) else None,
                    top_skus=[],
                    metrics={"timings_ms": {}, "counters": {}, "scores": {}, "labels": {}},
                    error=str(exc),
                )
            )

    summary = _aggregate(results)
    report_path = _report_path()
    _write_html_report(results, summary, report_path)

    failures = [
        f"{metric}={summary[metric]:.3f} < {threshold:.3f}"
        for metric, threshold in THRESHOLDS.items()
        if summary[metric] < threshold
    ]
    assert not failures, f"Metric evaluation failed; report written to {report_path}: {', '.join(failures)}"
