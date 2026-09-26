"""Esquema de puntuación: menos pistas usadas = más puntos."""
from __future__ import annotations

from pinpoint.config import NUM_CLUES


def points_for(clues_used: int, num_clues: int = NUM_CLUES) -> int:
    """Pista 1 -> num_clues puntos ... pista num_clues -> 1 punto."""
    if not 1 <= clues_used <= num_clues:
        raise ValueError(f"clues_used debe estar entre 1 y {num_clues}, se recibió {clues_used}")
    return num_clues - clues_used + 1


def max_points(rounds: int, num_clues: int = NUM_CLUES) -> int:
    return rounds * num_clues
