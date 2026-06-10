import pytest

from src.runtime.prompts import PromptNotFoundError, load_prompt


def test_load_prompt_reads_markdown() -> None:
    prompt = load_prompt("planner")

    assert "web_search_query" in prompt
    assert "English" in prompt


def test_load_prompt_fails_for_missing_prompt() -> None:
    with pytest.raises(PromptNotFoundError):
        load_prompt("missing-test-prompt")
