"""Comparación de respuestas y detección de pistas que revelan la respuesta."""
from __future__ import annotations

from collections.abc import Iterable

from nltk.metrics.distance import edit_distance

from pinpoint.config import LEAK_MIN_SUBSTRING, TYPO_MIN_LENGTH, TYPO_TOLERANCE
from pinpoint.models import Lang
from pinpoint.normalizer import clean, content_tokens, matches_expected_form, normalize


def is_correct(guess: str, answers: Iterable[str], lang: Lang) -> bool:
    """True si la respuesta del jugador equivale a alguna respuesta válida.

    Equivalencia = mismo conjunto de lemas/raíces. Como respaldo se tolera un
    error tipográfico (distancia de edición <= TYPO_TOLERANCE) en respuestas largas.
    """
    guess_norm = normalize(guess, lang)
    if not guess_norm:
        return False
    guess_surface = _surface(guess, lang)

    for answer in answers:
        if matches_expected_form(guess, answer, lang):
            return True
        answer_surface = _surface(answer, lang)
        if (
            TYPO_TOLERANCE > 0
            and len(answer_surface) >= TYPO_MIN_LENGTH
            and edit_distance(guess_surface, answer_surface) <= TYPO_TOLERANCE
        ):
            return True
    return False


def _surface(text: str, lang: Lang) -> str:
    """Forma limpia sin stopwords (sin stemming): base para medir errores tipográficos."""
    return " ".join(content_tokens(text, lang))


def leaks_answer(
    clue: str,
    answers: Iterable[str],
    lang: Lang,
    substring_answers: Iterable[str] | None = None,
) -> bool:
    """True si la pista revela la respuesta.

    - Raíz compartida con cualquier respuesta válida ('hunting dog' para 'dog').
    - Respuesta contenida como subcadena ('hotdog' para 'dog'). Por defecto se
      revisa contra todas las respuestas; substring_answers permite limitarlo
      (p. ej. solo a la forma canónica, para no descartar 'cánido' por el lema 'can').
    """
    answers = list(answers)
    clue_norm = normalize(clue, lang)
    clue_compact = clean(clue).replace(" ", "")

    if any(clue_norm & normalize(answer, lang) for answer in answers):
        return True

    for answer in answers if substring_answers is None else substring_answers:
        for token in content_tokens(answer, lang):
            if len(token) >= LEAK_MIN_SUBSTRING and token in clue_compact:
                return True
    return False
