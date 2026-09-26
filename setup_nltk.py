"""Descarga los recursos de NLTK que necesita el proyecto.

Uso:
    python setup_nltk.py

Primero usa los certificados del sistema. Si la descarga falla (típico en macOS con
Python de python.org: CERTIFICATE_VERIFY_FAILED), reintenta con los de `certifi`.
"""
from __future__ import annotations

import ssl
import sys
from collections.abc import Iterable

import nltk

# omw-1.4 lo usan versiones de NLTK < 3.9; omw-2.0 las más recientes.
RESOURCES: tuple[str, ...] = ("wordnet", "omw-1.4", "omw-2.0", "stopwords")
CRITICAL: frozenset[str] = frozenset({"wordnet", "stopwords"})


def download(resources: Iterable[str]) -> list[str]:
    """Descarga cada recurso y devuelve los que fallaron."""
    failed: list[str] = []
    for resource in resources:
        ok = nltk.download(resource, quiet=True, raise_on_error=False)
        print(f"[{'OK' if ok else 'ERROR'}] {resource}")
        if not ok:
            failed.append(resource)
    return failed


def use_certifi() -> bool:
    """Hace que las conexiones HTTPS usen el bundle de certifi. False si no está instalado."""
    try:
        import certifi
    except ImportError:
        return False
    cafile = certifi.where()
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)
    return True


def main() -> int:
    failed = download(RESOURCES)

    if failed:
        if use_certifi():
            print("Reintentando con los certificados de certifi...")
            failed = download(failed)
        else:
            print("[AVISO] certifi no está instalado: pip install certifi")

    # Verificación real: que el Open Multilingual WordNet responda en español.
    try:
        from nltk.corpus import wordnet as wn

        sample = wn.synset("dog.n.01").lemma_names("spa")
        print(f"[OK] OMW español disponible (dog.n.01 -> {sample[:3]}...)")
    except LookupError:
        print("[ERROR] WordNet/OMW no disponible.")
        print("  macOS con CERTIFICATE_VERIFY_FAILED: ejecuta 'Install Certificates.command'")
        print("  de tu versión de Python (en /Applications/Python 3.x/) y vuelve a intentar.")
        return 1

    # Basta con que una de las versiones de OMW se descargue (verificado arriba).
    return 1 if CRITICAL.intersection(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
