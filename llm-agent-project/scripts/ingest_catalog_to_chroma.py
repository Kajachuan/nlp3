from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore.chroma.catalog_store import ChromaCatalogStore


def main() -> None:
    catalog_path = PROJECT_ROOT / "data" / "raw" / "electronic_components_catalog.json"
    chroma_path = PROJECT_ROOT / "data" / "processed" / "chroma"
    store = ChromaCatalogStore(chroma_path)
    store.ensure_catalog_indexed(catalog_path)
    print(f"Indexed {store.collection.count()} catalog items into {chroma_path}")


if __name__ == "__main__":
    main()
