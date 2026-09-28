"""Normalización de texto: funciones de cadenas + lematización/stemming.

Pipeline (idéntico para respuesta esperada y respuesta del jugador):
    1. clean(): minúsculas, sin acentos, sin puntuación, espacios colapsados.
    2. tokenize(): separación por espacios.
    3. Eliminación de stopwords (artículos, preposiciones: "el perro", "a dog").
    4. Normalización morfológica por idioma:
       - EN: WordNetLemmatizer (irregulares: geese -> goose) + PorterStemmer.
       - ES: reducción acotada de diminutivos + SnowballStemmer.
       - FR: SnowballStemmer (NLTK no incluye lematizador para estos idiomas).
El resultado es un conjunto de raíces; comparar conjuntos hace que el orden de
las palabras no importe en respuestas multipalabra.
"""
from __future__ import annotations

import re
import string
import unicodedata
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from pinpoint.config import NLTK_LANG_NAME
from pinpoint.models import Lang

_SEPARATORS = re.compile(r"[_\-/]+")
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "¿¡«»“”‘’")

_lemmatizer = WordNetLemmatizer()
_porter = PorterStemmer()

# Snowball resuelve flexión de género y número, pero conserva los sufijos
# diminutivos. Se eliminan solo formas productivas frecuentes y siempre que quede
# una base de al menos tres caracteres (perrito -> perr, florecita -> flor).
_SPANISH_DIMINUTIVE_SUFFIXES = (
    "ecitos", "ecitas", "ecito", "ecita",
    "citos", "citas", "cito", "cita",
    "itos", "itas", "ito", "ita",
)


def strip_accents(text: str) -> str:
    """Elimina diacríticos: 'canción' -> 'cancion', 'ñandú' -> 'nandu'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean(text: str) -> str:
    """Limpieza con funciones de cadenas. No depende del idioma."""
    if not isinstance(text, str):
        raise TypeError(f"Se esperaba str, se recibió {type(text).__name__}")
    text = strip_accents(text.lower())
    text = _SEPARATORS.sub(" ", text)          # WordNet usa '_' en multipalabra
    text = text.translate(_PUNCT_TABLE)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return clean(text).split()


@lru_cache(maxsize=None)
def stopwords_for(lang: Lang) -> frozenset[str]:
    """Stopwords de NLTK ya limpiadas (sin acentos) para que coincidan con clean()."""
    return frozenset(clean(w) for w in stopwords.words(NLTK_LANG_NAME[lang]))


@lru_cache(maxsize=None)
def _snowball(lang: Lang) -> SnowballStemmer:
    return SnowballStemmer(NLTK_LANG_NAME[lang])


@lru_cache(maxsize=4096)
def normalize_token(token: str, lang: Lang) -> str:
    if lang is Lang.EN:
        return _porter.stem(_lemmatizer.lemmatize(token, pos="n"))
    if lang is Lang.ES:
        token = _strip_spanish_diminutive(token)
    return _snowball(lang).stem(token)


def _strip_spanish_diminutive(token: str) -> str:
    """Reduce diminutivos españoles comunes antes de aplicar Snowball."""
    for suffix in _SPANISH_DIMINUTIVE_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def content_tokens(text: str, lang: Lang) -> list[str]:
    """Tokens sin stopwords. Si todos son stopwords, se conservan (evita vacío)."""
    tokens = tokenize(text)
    filtered = [t for t in tokens if t not in stopwords_for(lang)]
    return filtered or tokens


def normalize(text: str, lang: Lang) -> frozenset[str]:
    """Forma canónica comparable: conjunto de lemas/raíces de las palabras de contenido."""
    return frozenset(normalize_token(t, lang) for t in content_tokens(text, lang))


def canonical_key(text: str, lang: Lang) -> str:
    """Representación en cadena de normalize(), útil para deduplicar y medir distancias."""
    return " ".join(sorted(normalize(text, lang)))
