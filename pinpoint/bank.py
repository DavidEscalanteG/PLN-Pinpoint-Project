"""Carga y validación estructural del banco de categorías (data/categories.json)."""
from __future__ import annotations

import json
from pathlib import Path

from pinpoint.config import CATEGORIES_PATH
from pinpoint.models import Category, CategoryBankError


def _parse_entry(entry: object, index: int) -> Category:
    if not isinstance(entry, dict) or not isinstance(entry.get("synset"), str):
        raise CategoryBankError(f"Entrada #{index}: se esperaba un objeto con clave 'synset'")
    display = entry.get("display", {})
    if not isinstance(display, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in display.items()
    ):
        raise CategoryBankError(f"Entrada #{index} ({entry['synset']}): 'display' inválido")
    exclude = entry.get("exclude", [])
    if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
        raise CategoryBankError(f"Entrada #{index} ({entry['synset']}): 'exclude' debe ser lista de synsets")
    return Category(
        synset_id=entry["synset"].strip(),
        display={k: v.strip() for k, v in display.items()},
        exclude=frozenset(x.strip() for x in exclude),
    )


def load_categories(path: Path = CATEGORIES_PATH) -> list[Category]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CategoryBankError(f"No existe el banco de categorías: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CategoryBankError(f"JSON inválido en {path}: {exc}") from exc

    entries = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CategoryBankError("El banco debe tener la forma {\"categories\": [...]}")

    categories: list[Category] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        category = _parse_entry(entry, i)
        if category.synset_id not in seen:
            seen.add(category.synset_id)
            categories.append(category)

    if not categories:
        raise CategoryBankError("El banco de categorías está vacío")
    return categories


def save_categories(categories: list[Category], path: Path = CATEGORIES_PATH) -> None:
    payload = {
        "version": 1,
        "categories": [
            {
                "synset": c.synset_id,
                **({"display": c.display} if c.display else {}),
                **({"exclude": sorted(c.exclude)} if c.exclude else {}),
            }
            for c in categories
        ],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
