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


# Construye una ronda con sus pistas y respuestas válidas.
def build_round(categoria: Category, lang: Lang, pistas: int = NUM_CLUES) -> Round:
    display = display_answer(categoria, lang)
    clues = generate_clues(categoria.synset_id, lang, display, pistas, categoria.exclude)
    shown_synonyms = [c.text for c in clues if c.relation is Relation.SYNONYM]
    return Round(
        target_synset=categoria.synset_id,
        lang=lang,
        display_answer=display,
        valid_answers=valid_answers(categoria.synset_id, lang, display, exclude=shown_synonyms),
        clues=clues,
    )


# Gestiona el estado de una partida y sus rondas.
class GameSession:


    # Inicializa una nueva sesión de juego con las categorías disponibles.
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

    # Devuelve la ronda actual.
    @property
    def round(self) -> Round | None:
        return self._round

    # Devuelve la cantidad de rondas completadas.
    @property
    def rounds_played(self) -> int:
        return len(self.summary.results)

    # Indica si todas las rondas de la partida han terminado.
    @property
    def is_over(self) -> bool:
        return self.rounds_played >= self.total_rounds

    # Indica si existe una ronda activa en este momento.
    @property
    def round_active(self) -> bool:
        return self._round is not None and not self._round_finished

    # Crea y comienza una nueva ronda.
    def new_round(self) -> Round:
        if self.is_over:
            raise PinpointError("La partida ya terminó")
        if self.round_active:
            raise PinpointError("Hay una ronda en curso")

        while self._queue:
            categoria = self._queue.pop()
            try:
                built = build_round(categoria, self.lang)
            except InsufficientCluesError as exc:
                logger.warning("Categoría omitida: %s", exc)
                continue
            self._round, self._clue_idx, self._round_finished = built, 0, False
            return built
        raise CategoryBankError("No quedan categorías jugables en el banco")

    # Devuelve la pista que corresponde mostrar actualmente.
    def current_clue(self) -> Clue:
        return self._require_round().clues[self._clue_idx]

    # Devuelve todas las pistas reveladas hasta el momento.
    def revealed_clues(self) -> list[Clue]:
        return self._require_round().clues[: self._clue_idx + 1]

    # Comprueba la respuesta del jugador y avanza la ronda si es incorrecta.
    def guess(self, text: str) -> GuessResult:
        current = self._require_round()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("La respuesta no puede estar vacía")
        if is_correct(text, current.valid_answers, self.lang):
            return self._finish(correct=True)
        return self._advance()

    # Pasa la pista actual y avanza hacia la siguiente.
    def pass_turn(self) -> GuessResult:
        self._require_round()
        return self._advance()

    # Verifica que exista una ronda activa y devuelve su información.
    def _require_round(self) -> Round:
        if self._round is None or self._round_finished:
            raise PinpointError("No hay una ronda activa; llama a new_round()")
        return self._round

    # Avanza a la siguiente pista o finaliza la ronda si no quedan pistas.
    def _advance(self) -> GuessResult:
        clues_used = self._clue_idx + 1
        if clues_used >= len(self._round.clues):  
            return self._finish(correct=False)
        self._clue_idx += 1
        return GuessResult(correct=False, clues_used=clues_used, finished=False)

    # Finaliza la ronda, registra el resultado y calcula los puntos obtenidos.
    def _finish(self, correct: bool) -> GuessResult:
        clues_used = self._clue_idx + 1
        result = GuessResult(
            correct=correct,
            clues_used=clues_used,
            finished=True,
            points=points_for(clues_used, len(self._round.clues)) if correct else 0, 
            revealed_answer=self._round.display_answer,  
        )
        self.summary.results.append(result)
        self._round_finished = True
        return result