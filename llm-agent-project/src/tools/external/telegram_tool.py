from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import parse, request

from langchain_core.tools import tool


@dataclass(frozen=True)
class TelegramSendResult:
    """Result returned by the Telegram notification tool."""

    ok: bool
    message: str
    chat_id: str | None = None
    error: str | None = None


def build_purchase_message(product_name: str, price: str, link: str) -> str:
    """Build the customer-facing Telegram purchase message."""
    clean_price = price.strip()
    if clean_price.upper().startswith("USD"):
        clean_price = clean_price[3:].strip()
    return f"Puedes comprar {product_name.strip()} por USD {clean_price} en {link.strip()}"


def send_telegram_purchase_message(
    product_name: str,
    price: str,
    link: str,
    bot_api_key: str | None = None,
    client_id: str | None = None,
    opener: Any | None = None,
) -> TelegramSendResult:
    """Send a purchase message to the configured Telegram client."""
    token = (bot_api_key or os.getenv("TELEGRAM_BOT_API_KEY") or "").strip()
    chat_id = (client_id or os.getenv("TELEGRAM_CLIENT_ID") or "").strip()
    if not token:
        return TelegramSendResult(ok=False, message="", error="TELEGRAM_BOT_API_KEY is not configured")
    if not chat_id:
        return TelegramSendResult(ok=False, message="", error="TELEGRAM_CLIENT_ID is not configured")
    if not product_name or not price or not link:
        return TelegramSendResult(ok=False, message="", error="product_name, price and link are required")

    message = build_purchase_message(product_name, price, link)
    payload = parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    http_open = opener or request.urlopen
    try:
        with http_open(request.Request(url, data=payload, method="POST"), timeout=10) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return TelegramSendResult(ok=False, message=message, chat_id=chat_id, error=str(exc))

    if not bool(response_payload.get("ok")):
        return TelegramSendResult(
            ok=False,
            message=message,
            chat_id=chat_id,
            error=str(response_payload.get("description") or response_payload),
        )
    return TelegramSendResult(ok=True, message=message, chat_id=chat_id)


@tool("send_telegram_message")
def send_telegram_message(product_name: str, price: str, link: str) -> dict[str, Any]:
    """Send a Telegram purchase message to the configured customer."""
    result = send_telegram_purchase_message(product_name=product_name, price=price, link=link)
    return result.__dict__
