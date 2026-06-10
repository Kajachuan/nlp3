from src.runtime.config import DEFAULT_MODEL_OPTIONS, parse_model_options


def test_parse_model_options_from_csv() -> None:
    assert parse_model_options(" groq/compound-mini, , llama-3.3-70b-versatile ") == [
        "groq/compound-mini",
        "llama-3.3-70b-versatile",
    ]


def test_parse_model_options_uses_defaults_when_empty() -> None:
    assert parse_model_options("") == DEFAULT_MODEL_OPTIONS
