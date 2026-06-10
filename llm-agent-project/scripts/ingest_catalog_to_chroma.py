from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.chroma.catalog_store import ChromaCatalogStore
from src.runtime.catalog_data import load_catalog_items


def main() -> None:
    catalog_path = PROJECT_ROOT / "data" / "raw"
    chroma_path = PROJECT_ROOT / "data" / "processed" / "chroma"
    catalog_items = load_catalog_items(catalog_path)
    store = ChromaCatalogStore(chroma_path)
    store.ensure_items_indexed(catalog_items)
    print(f"Indexed {store.collection.count()} catalog items from {catalog_path} into {chroma_path}")


if __name__ == "__main__":
    main()
