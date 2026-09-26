"""Pinpoint WordNet: juego de adivinanza de categorías con pistas generadas desde WordNet."""
from pinpoint.game import GameSession
from pinpoint.models import Clue, GuessResult, Lang, Relation, Round

__all__ = ["GameSession", "Clue", "GuessResult", "Lang", "Relation", "Round"]
__version__ = "1.0.0"
