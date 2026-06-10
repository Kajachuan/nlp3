from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_catalog_items(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    items: list[dict[str, Any]] = []
    seen_skus: set[str] = set()

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Catalog file must contain a JSON list: {file_path}")
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Catalog item must be an object in {file_path}")
            sku = str(item.get("sku", "")).strip()
            if not sku:
                raise ValueError(f"Catalog item without sku in {file_path}")
            if sku in seen_skus:
                raise ValueError(f"Duplicate catalog sku '{sku}' found in {file_path}")
            seen_skus.add(sku)
            items.append(item)

    return items
