"""Capa bilingüe: convierte synsets en palabras del idioma elegido vía Open Multilingual WordNet."""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, WordNetError

from pinpoint.models import Category, CategoryBankError, InsufficientCluesError, Lang, RawClue, Relation


def get_synset(synset_id: str) -> Synset:
    try:
        return wn.synset(synset_id)
    except (WordNetError, ValueError) as exc:
        raise CategoryBankError(f"Synset inexistente en WordNet: '{synset_id}'") from exc


@lru_cache(maxsize=None)
def lemmas_in(synset_id: str, lang: Lang) -> tuple[str, ...]:
    """Lemas del synset en el idioma dado, legibles ('_' -> ' ') y sin duplicados."""
    synset = get_synset(synset_id)
    names = synset.lemma_names() if lang is Lang.EN else synset.lemma_names(lang.value)
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        text = name.replace("_", " ").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


def has_lemmas(synset_id: str, lang: Lang) -> bool:
    return bool(lemmas_in(synset_id, lang))


def _is_acronym(form: str) -> bool:
    compact = form.replace(".", "")
    return compact.isupper() and len(compact) <= 5


def rank_forms(forms: Iterable[str]) -> list[str]:
    """Ordena formas por legibilidad: sin dígitos > sin siglas > una sola palabra.

    El ordenamiento es estable, así que en inglés se conserva el orden de WordNet
    (aproximadamente por frecuencia de uso).
    """
    return sorted(
        forms,
        key=lambda f: (any(ch.isdigit() for ch in f), _is_acronym(f), len(f.split()) > 1),
    )


def display_answer(category: Category, lang: Lang) -> str:
    """Forma canónica de la respuesta: override del banco o primer lema de WordNet."""
    override = category.display.get(lang.value)
    if override:
        return override
    forms = rank_forms(lemmas_in(category.synset_id, lang))
    if not forms:
        raise InsufficientCluesError(f"{category.synset_id} no tiene lemas en {lang.label}")
    return forms[0]


def surface_forms(raw: RawClue, target_synset: str, lang: Lang, display: str) -> list[str]:
    """Palabras candidatas para mostrar una pista, en orden de preferencia."""
    if raw.relation is Relation.SYNONYM:
        forms = [f for f in lemmas_in(target_synset, lang) if f.lower() != display.lower()]
    else:
        forms = list(lemmas_in(raw.synset_id, lang))
    return rank_forms(forms)


def valid_answers(
    target_synset: str, lang: Lang, display: str, exclude: Iterable[str] = ()
) -> list[str]:
    """Respuestas aceptadas: la forma canónica + los demás lemas del synset.

    Se excluyen los sinónimos que ya se mostraron como pista (escribir la pista
    no cuenta como adivinar).
    """
    excluded = {e.lower() for e in exclude}
    others = [
        f for f in lemmas_in(target_synset, lang)
        if f.lower() != display.lower() and f.lower() not in excluded
    ]
    return [display, *others]
