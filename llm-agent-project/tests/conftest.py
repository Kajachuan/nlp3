from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-eval-metrics",
        action="store_true",
        default=False,
        help="Run dataset-driven metric evaluation tests that call the real pipeline.",
    )
    parser.addoption(
        "--run-deepeval-metrics",
        action="store_true",
        default=False,
        help="Run DeepEval-based metric evaluation tests that call the real pipeline.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_metrics = config.getoption("--run-eval-metrics")
    run_deepeval_metrics = config.getoption("--run-deepeval-metrics")
    skip_eval = pytest.mark.skip(reason="need --run-eval-metrics option to run metric evaluation")
    skip_deepeval = pytest.mark.skip(reason="need --run-deepeval-metrics option to run DeepEval metric evaluation")
    for item in items:
        if "eval_metrics" in item.keywords and not run_eval_metrics:
            item.add_marker(skip_eval)
        if "eval_deepeval" in item.keywords and not run_deepeval_metrics:
            item.add_marker(skip_deepeval)
