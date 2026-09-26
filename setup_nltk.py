"""Descarga los recursos de NLTK que necesita el proyecto.

Uso:
    python setup_nltk.py
"""
from __future__ import annotations

import sys

import nltk

# omw-1.4 lo usan versiones de NLTK < 3.9; omw-2.0 las más recientes.
RESOURCES: tuple[str, ...] = ("wordnet", "omw-1.4", "omw-2.0", "stopwords")


def main() -> int:
    failed: list[str] = []
    for resource in RESOURCES:
        ok = nltk.download(resource, quiet=True)
        print(f"[{'OK' if ok else 'ERROR'}] {resource}")
        if not ok:
            failed.append(resource)

    # Verificación real: que el Open Multilingual WordNet responda en español.
    try:
        from nltk.corpus import wordnet as wn

        sample = wn.synset("dog.n.01").lemma_names("spa")
        print(f"[OK] OMW español disponible (dog.n.01 -> {sample[:3]}...)")
    except LookupError as exc:
        print(f"[ERROR] OMW no disponible: {exc}")
        return 1

    # Basta con que una de las versiones de OMW se descargue (verificado arriba).
    critical = {"wordnet", "stopwords"}
    return 1 if critical.intersection(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
