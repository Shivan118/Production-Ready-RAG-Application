"""Golden evaluation dataset loader."""

import json
from functools import lru_cache
from pathlib import Path

from app.models.schemas import GoldenItem

DATASET_PATH = Path("evals/golden_dataset.json")


@lru_cache
def _load() -> tuple[str, list[GoldenItem]]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    items = [GoldenItem(**item) for item in data["items"]]
    return data.get("description", ""), items


def description() -> str:
    return _load()[0]


def golden_items(limit: int | None = None) -> list[GoldenItem]:
    items = _load()[1]
    return items[:limit] if limit else items


def size() -> int:
    return len(_load()[1])
