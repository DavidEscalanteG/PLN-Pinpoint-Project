"""Constantes de configuración del juego."""
from __future__ import annotations

from pathlib import Path

from pinpoint.models import Lang, Relation

ROOT_DIR = Path(__file__).resolve().parent.parent
CATEGORIES_PATH = ROOT_DIR / "data" / "categories.json"

NUM_CLUES = 5
DEFAULT_ROUNDS = 3

# Plan de pistas: cada posición lista la relación preferida y sus respaldos, en orden.
# Pista 1 = más general, pista 5 = más reveladora.
CLUE_PLAN: tuple[tuple[Relation, ...], ...] = (
    (Relation.HYPERNYM_FAR, Relation.HYPERNYM, Relation.SIBLING),
    (Relation.HYPERNYM, Relation.SIBLING, Relation.HYPONYM),
    (Relation.HYPONYM, Relation.SIBLING),
    (Relation.HYPONYM, Relation.SIBLING, Relation.SYNONYM),
    (Relation.SYNONYM, Relation.HYPONYM, Relation.SIBLING),
)

# Distancia (en niveles de hiperonimia) de los candidatos a la pista 1.
# Se elige el ancestro conocido más cercano dentro de este rango.
FAR_HYPERNYM_MIN_LEVEL = 2
FAR_HYPERNYM_MAX_LEVEL = 5

# Hiperónimos demasiado abstractos para servir como pista.
GENERIC_SYNSETS: frozenset[str] = frozenset({
    "entity.n.01", "physical_entity.n.01", "abstraction.n.06", "object.n.01",
    "whole.n.02", "thing.n.12", "matter.n.03", "causal_agent.n.01",
    "living_thing.n.01", "organism.n.01", "artifact.n.01", "instrumentality.n.03",
    "unit.n.03", "group.n.01", "psychological_feature.n.01", "attribute.n.02",
    "solid.n.01", "substance.n.01", "substance.n.07", "part.n.02", "relation.n.01",
})

# Nombres de idioma para stopwords y SnowballStemmer de NLTK.
NLTK_LANG_NAME: dict[Lang, str] = {
    Lang.EN: "english",
    Lang.ES: "spanish",
    Lang.FR: "french",
}

# Tolerancia a errores tipográficos (distancia de edición sobre la forma normalizada).
TYPO_TOLERANCE = 1
TYPO_MIN_LENGTH = 6  # reduce falsos positivos en palabras cortas (p. ej. "barco"/"banco")

# Longitud mínima de un token de la respuesta para detectarlo como subcadena en una pista.
LEAK_MIN_SUBSTRING = 3
