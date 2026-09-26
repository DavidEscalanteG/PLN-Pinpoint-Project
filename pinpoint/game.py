"""Lógica del juego: construcción de rondas, revelado progresivo y control de intentos.

No hace I/O: la CLI, un notebook o una web consumen esta clase directamente.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Sequence

from pinpoint.bank import load_categories
from pinpoint.config import DEFAULT_ROUNDS, NUM_CLUES
from pinpoint.language import display_answer, valid_answers
from pinpoint.matcher import is_correct
from pinpoint.models import (
    Category,
    CategoryBankError,
    Clue,
    GuessResult,
    InsufficientCluesError,
    Lang,
    MatchSummary,
    PinpointError,
    Relation,
    Round,
)
from pinpoint.scoring import points_for
from pinpoint.wordnet_clues import generate_clues

logger = logging.getLogger(__name__)


def build_round(category: Category, lang: Lang, num_clues: int = NUM_CLUES) -> Round:
    """Genera una ronda completa (pistas + respuestas válidas) para una categoría."""
    display = display_answer(category, lang)
    clues = generate_clues(category.synset_id, lang, display, num_clues, category.exclude)
    shown_synonyms = [c.text for c in clues if c.relation is Relation.SYNONYM]
    return Round(
        target_synset=category.synset_id,
        lang=lang,
        display_answer=display,
        valid_answers=valid_answers(category.synset_id, lang, display, exclude=shown_synonyms),
        clues=clues,
    )


class GameSession:
    """Partida de varias rondas.

    Flujo:
        session = GameSession(Lang.ES, rounds=3)
        while not session.is_over:
            session.new_round()
            while session.round_active:
                clue = session.current_clue()
                result = session.guess(input())
    """

    def __init__(
        self,
        lang: Lang | str,
        rounds: int = DEFAULT_ROUNDS,
        seed: int | None = None,
        categories: Sequence[Category] | None = None,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds debe ser >= 1")
        self.lang = Lang(lang)
        pool = list(categories) if categories is not None else load_categories()
        if not pool:
            raise CategoryBankError("No hay categorías disponibles")

        self._rng = random.Random(seed)
        self._rng.shuffle(pool)
        self._queue: list[Category] = pool
        self.total_rounds = min(rounds, len(pool))
        self.summary = MatchSummary()
        self._round: Round | None = None
        self._clue_idx = 0
        self._round_finished = True

    # ---------- Estado ----------
    @property
    def round(self) -> Round | None:
        return self._round

    @property
    def rounds_played(self) -> int:
        return len(self.summary.results)

    @property
    def is_over(self) -> bool:
        return self.rounds_played >= self.total_rounds

    @property
    def round_active(self) -> bool:
        return self._round is not None and not self._round_finished

    # ---------- Acciones ----------
    def new_round(self) -> Round:
        """Selecciona la siguiente categoría jugable y genera sus pistas."""
        if self.is_over:
            raise PinpointError("La partida ya terminó")
        if self.round_active:
            raise PinpointError("Hay una ronda en curso")

        while self._queue:
            category = self._queue.pop()
            try:
                built = build_round(category, self.lang)
            except InsufficientCluesError as exc:
                logger.warning("Categoría omitida: %s", exc)
                continue
            self._round, self._clue_idx, self._round_finished = built, 0, False
            return built
        raise CategoryBankError("No quedan categorías jugables en el banco")

    def current_clue(self) -> Clue:
        return self._require_round().clues[self._clue_idx]

    def revealed_clues(self) -> list[Clue]:
        return self._require_round().clues[: self._clue_idx + 1]

    def guess(self, text: str) -> GuessResult:
        """Evalúa una respuesta. Si falla, revela la siguiente pista o termina la ronda."""
        current = self._require_round()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("La respuesta no puede estar vacía")
        if is_correct(text, current.valid_answers, self.lang):
            return self._finish(correct=True)
        return self._advance()

    def pass_turn(self) -> GuessResult:
        """Salta a la siguiente pista sin responder (cuenta como intento fallido)."""
        self._require_round()
        return self._advance()

    # ---------- Internos ----------
    def _require_round(self) -> Round:
        if self._round is None or self._round_finished:
            raise PinpointError("No hay una ronda activa; llama a new_round()")
        return self._round

    def _advance(self) -> GuessResult:
        clues_used = self._clue_idx + 1
        if clues_used >= len(self._round.clues):  # type: ignore[union-attr]
            return self._finish(correct=False)
        self._clue_idx += 1
        return GuessResult(correct=False, clues_used=clues_used, finished=False)

    def _finish(self, correct: bool) -> GuessResult:
        clues_used = self._clue_idx + 1
        result = GuessResult(
            correct=correct,
            clues_used=clues_used,
            finished=True,
            points=points_for(clues_used, len(self._round.clues)) if correct else 0,  # type: ignore[union-attr]
            revealed_answer=self._round.display_answer,  # type: ignore[union-attr]
        )
        self.summary.results.append(result)
        self._round_finished = True
        return result
