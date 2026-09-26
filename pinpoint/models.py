"""Contratos de datos y excepciones compartidos entre módulos."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Lang(str, Enum):
    """Códigos de idioma del Open Multilingual WordNet."""

    EN = "eng"
    ES = "spa"
    FR = "fra"

    @property
    def label(self) -> str:
        return {"eng": "Inglés", "spa": "Español", "fra": "Francés"}[self.value]


class Relation(str, Enum):
    """Relación semántica de WordNet de la que proviene una pista."""

    HYPERNYM_L2 = "hiperónimo (2 niveles)"
    HYPERNYM = "hiperónimo"
    SIBLING = "co-hipónimo"
    HYPONYM = "hipónimo"
    SYNONYM = "sinónimo"


@dataclass(frozen=True)
class Category:
    """Entrada del banco de categorías.

    display: forma preferida de la respuesta por idioma (clave = código OMW).
             Si falta un idioma, se usa el primer lema de WordNet en ese idioma.
    exclude: synsets que nunca deben usarse como pista (curaduría manual de ruido de OMW).
    """

    synset_id: str
    display: dict[str, str] = field(default_factory=dict)
    exclude: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RawClue:
    """Pista independiente del idioma: synset de origen + relación semántica."""

    synset_id: str
    relation: Relation


@dataclass(frozen=True)
class Clue:
    """Pista renderizada en un idioma concreto."""

    order: int  # 1 = más general/difícil, 5 = más específica/reveladora
    text: str
    relation: Relation
    synset_id: str


@dataclass
class Round:
    target_synset: str
    lang: Lang
    display_answer: str
    valid_answers: list[str]
    clues: list[Clue]


@dataclass(frozen=True)
class GuessResult:
    correct: bool
    clues_used: int
    finished: bool
    points: int = 0
    revealed_answer: str | None = None


@dataclass
class MatchSummary:
    results: list[GuessResult] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(r.points for r in self.results)

    @property
    def rounds_won(self) -> int:
        return sum(1 for r in self.results if r.correct)


class PinpointError(Exception):
    """Error base del proyecto."""


class InsufficientCluesError(PinpointError):
    """No se pudieron generar suficientes pistas válidas para un synset."""


class CategoryBankError(PinpointError):
    """Banco de categorías inválido, vacío o agotado."""
