"""Esquema de puntuación: menos pistas usadas = más puntos."""
from __future__ import annotations

from pinpoint.config import NUM_CLUES


def points_for(pistas_usadas: int, pistas: int = NUM_CLUES) -> int:
    #Calcula los puntos obtenidos
    if not 1 <= pistas_usadas <= pistas:
        raise ValueError(f"pistas_usadas debe estar entre 1 y {pistas}, se recibió {pistas_usadas}")
    return pistas - pistas_usadas + 1


def max_points(rondas: int, pistas: int = NUM_CLUES) -> int:
    #Calcula el puntaje máximo posible
    return rondas * pistas
