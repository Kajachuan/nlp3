from pathlib import Path

from src.agents.orchestration.component_advisor import ComponentAdvisorOrchestrator


def _orchestrator() -> ComponentAdvisorOrchestrator:
    return ComponentAdvisorOrchestrator(
        local_hybrid_agent=None,  # type: ignore[arg-type]
        web_agent=None,  # type: ignore[arg-type]
        evidence_gate=None,  # type: ignore[arg-type]
        audit_log_path=Path("unused.jsonl"),
    )


def test_removes_accidental_markdown_fences_from_answer() -> None:
    answer = """```markdown
Componente: ACS71240

Precio: USD 2.83
Comprar: [Click aqui](https://example.com/product)
```"""

    cleaned = _orchestrator()._clean_markdown_fences(answer)

    assert cleaned == (
        "Componente: ACS71240\n\n"
        "Precio: USD 2.83\n"
        "Comprar: [Click aqui](https://example.com/product)"
    )


def test_keeps_unfenced_markdown_answer_unchanged() -> None:
    answer = "Comprar: [Click aqui](https://example.com/product)"

    assert _orchestrator()._clean_markdown_fences(answer) == answer


def test_percent_encodes_spaces_inside_markdown_links() -> None:
    answer = (
        "Precio: 2.15\n"
        "Comprar: [Click aqui](https://www.google.com/search?ibp=oshop&q=A1468 sensor&udm=28)"
    )

    cleaned = _orchestrator()._clean_markdown_fences(answer)

    assert cleaned == (
        "Precio: 2.15\n"
        "Comprar: [Click aqui](https://www.google.com/search?ibp=oshop&q=A1468%20sensor&udm=28)"
    )


def test_percent_encodes_parentheses_inside_markdown_links() -> None:
    answer = "Comprar: [Click aqui](https://example.com/search?q=opamp (low noise))"

    cleaned = _orchestrator()._clean_markdown_fences(answer)

    assert cleaned == "Comprar: [Click aqui](https://example.com/search?q=opamp%20%28low%20noise%29)"
