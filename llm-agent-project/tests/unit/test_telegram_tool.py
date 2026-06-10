from __future__ import annotations

from urllib import parse

from src.tools.external.telegram_tool import build_purchase_message, send_telegram_purchase_message


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return b'{"ok": true, "result": {"message_id": 1}}'


def test_build_purchase_message_normalizes_usd_prefix() -> None:
    message = build_purchase_message("AMS1117", "USD 0.42", "catalog://REG-AMS1117-3V3")

    assert message == "Puedes comprar AMS1117 por USD 0.42 en catalog://REG-AMS1117-3V3"


def test_send_telegram_purchase_message_posts_expected_payload() -> None:
    captured = {}

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["data"] = parse.parse_qs(request.data.decode("utf-8"))
        return FakeResponse()

    result = send_telegram_purchase_message(
        product_name="AMS1117",
        price="0.42",
        link="catalog://REG-AMS1117-3V3",
        bot_api_key="token",
        client_id="123",
        opener=fake_opener,
    )

    assert result.ok is True
    assert result.chat_id == "123"
    assert captured["url"] == "https://api.telegram.org/bottoken/sendMessage"
    assert captured["timeout"] == 10
    assert captured["data"]["chat_id"] == ["123"]
    assert captured["data"]["text"] == [
        "Puedes comprar AMS1117 por USD 0.42 en catalog://REG-AMS1117-3V3"
    ]


def test_send_telegram_purchase_message_requires_config(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_CLIENT_ID", raising=False)

    result = send_telegram_purchase_message("AMS1117", "0.42", "catalog://REG-AMS1117-3V3")

    assert result.ok is False
    assert result.error == "TELEGRAM_BOT_API_KEY is not configured"
