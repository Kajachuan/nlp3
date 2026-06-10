from src.tools.external.web_search_adapter import WebSearchAdapter


class FakeWebSearchClient:
    def search(self, **kwargs):
        assert kwargs["method"] == "tavily"
        assert kwargs["preferred_sites"] == ["digikey.com"]
        assert kwargs["include_delivery_details"] is False
        return [
            {
                "title": "BME280 breakout",
                "url": "https://digikey.com/example",
                "content": "Sensor ambiental con I2C.",
                "score": 0.91,
            }
        ]


def test_adapter_normalizes_real_tool_results(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    adapter = WebSearchAdapter(method="tavily")
    adapter.client = FakeWebSearchClient()

    results = adapter.search_with_options(
        "bme280 i2c",
        limit=1,
        method="tavily",
        preferred_sites=["digikey.com"],
    )

    assert adapter.last_mode == "web_search_tool"
    assert results[0].title == "BME280 breakout"
    assert results[0].url == "https://digikey.com/example"
    assert results[0].score == 0.91
