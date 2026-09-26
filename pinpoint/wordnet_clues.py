"""Generación de pistas a partir de relaciones de WordNet.

Relaciones explotadas (ver config.CLUE_PLAN para el orden):
    - Hiperónimos a 2 niveles y directos   -> pistas generales (1-2)
    - Hipónimos (por frecuencia de uso)    -> pistas específicas (3-4)
    - Sinónimos del mismo synset           -> pista más reveladora (5)
    - Co-hipónimos (hermanos)              -> respaldo cuando falta alguna relación
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from nltk.corpus.reader.wordnet import Synset

from pinpoint.config import CLUE_PLAN, GENERIC_SYNSETS, NUM_CLUES
from pinpoint.language import get_synset, has_lemmas, lemmas_in, surface_forms
from pinpoint.matcher import leaks_answer
from pinpoint.models import Clue, InsufficientCluesError, Lang, RawClue, Relation
from pinpoint.normalizer import normalize


def _hypernyms(synset: Synset) -> list[Synset]:
    return synset.hypernyms() + synset.instance_hypernyms()


def _hyponyms(synset: Synset) -> list[Synset]:
    return synset.hyponyms() + synset.instance_hyponyms()


def _popularity(synset: Synset) -> int:
    """Frecuencia en SemCor (solo inglés); aproxima qué tan conocida es la palabra."""
    return sum(lemma.count() for lemma in synset.lemmas())


def _unique(synsets: Iterable[Synset], exclude: set[str]) -> list[Synset]:
    seen = set(exclude)
    result: list[Synset] = []
    for s in synsets:
        name = s.name()
        if name not in seen and name not in GENERIC_SYNSETS:
            seen.add(name)
            result.append(s)
    return result


def _is_proper_noun(synset: Synset) -> bool:
    """Nombres propios (cultivares, marcas, personas) suelen ser pistas oscuras."""
    return synset.lemma_names()[0][:1].isupper()


def _by_popularity(synsets: list[Synset]) -> list[Synset]:
    """Nombres comunes primero; luego más frecuentes; el nombre desempata (determinismo)."""
    return sorted(synsets, key=lambda s: (_is_proper_noun(s), -_popularity(s), s.name()))


@lru_cache(maxsize=None)
def candidate_pools(synset_id: str) -> dict[Relation, tuple[RawClue, ...]]:
    """Candidatos por relación, independientes del idioma y ordenados por preferencia."""
    target = get_synset(synset_id)
    parents = _by_popularity(_hypernyms(target))
    grandparents = _by_popularity([g for p in parents for g in _hypernyms(p)])
    children = _by_popularity(_unique(_hyponyms(target), {synset_id}))
    grandchildren = _by_popularity(_unique((g for c in children for g in _hyponyms(c)), {synset_id}))
    siblings = _by_popularity(_unique((s for p in parents for s in _hyponyms(p)), {synset_id}))

    def raw(synsets: Iterable[Synset], relation: Relation) -> tuple[RawClue, ...]:
        return tuple(RawClue(s.name(), relation) for s in synsets)

    parent_names = {p.name() for p in parents}
    return {
        Relation.HYPERNYM_L2: raw(_unique(grandparents, parent_names | {synset_id}), Relation.HYPERNYM_L2),
        Relation.HYPERNYM: raw(_unique(parents, {synset_id}), Relation.HYPERNYM),
        # Hipónimos directos primero; los de segundo nivel amplían el banco de candidatos.
        Relation.HYPONYM: raw(children, Relation.HYPONYM)
        + raw([g for g in grandchildren if g not in children], Relation.HYPONYM),
        Relation.SIBLING: raw(siblings, Relation.SIBLING),
        Relation.SYNONYM: (RawClue(synset_id, Relation.SYNONYM),),
    }


class _ClueSelector:
    """Elige pistas sin repetir synsets ni palabras y sin revelar la respuesta."""

    def __init__(self, target: str, lang: Lang, display: str, exclude: frozenset[str]) -> None:
        self.target = target
        self.lang = lang
        self.display = display
        self.answers = [display, *lemmas_in(target, lang)]
        self.pools = candidate_pools(target)
        self.used_synsets: set[str] = {target, *exclude}
        self.used_stems: set[str] = set()

    def _acceptable(self, text: str, relation: Relation) -> bool:
        stems = normalize(text, self.lang)
        if not stems or stems & self.used_stems:  # repetida o muy parecida a otra pista
            return False
        # Un sinónimo es, por definición, una respuesta válida: se compara contra las demás.
        answers = (
            [a for a in self.answers if a.lower() != text.lower()]
            if relation is Relation.SYNONYM
            else self.answers
        )
        return not leaks_answer(text, answers, self.lang, substring_answers=[self.display])

    def _untranslated(self, text: str, synset_id: str) -> bool:
        """OMW a veces conserva el lema en inglés ('ice bear' en español)."""
        if self.lang is Lang.EN:
            return False
        return text.lower() in {l.lower() for l in lemmas_in(synset_id, Lang.EN)}

    def pick(self, order: int, relations: tuple[Relation, ...]) -> Clue | None:
        # 1.ª pasada: solo nombres comunes. 2.ª pasada: se permiten nombres propios/siglas.
        for allow_proper in (False, True):
            for relation in relations:
                clue = self._pick_from(order, relation, allow_proper)
                if clue is not None:
                    return clue
        return None

    def _pick_from(self, order: int, relation: Relation, allow_proper: bool) -> Clue | None:
        is_synonym = relation is Relation.SYNONYM
        for raw in self.pools[relation]:
            if not is_synonym and (
                raw.synset_id in self.used_synsets or not has_lemmas(raw.synset_id, self.lang)
            ):
                continue
            for text in surface_forms(raw, self.target, self.lang, self.display):
                # 1.ª pasada: se evitan nombres propios y palabras que OMW dejó sin traducir.
                # Los sinónimos pueden ser nombres científicos ("Canis familiaris"): se permiten siempre.
                if not allow_proper and not is_synonym and (
                    text[:1].isupper() or self._untranslated(text, raw.synset_id)
                ):
                    continue
                if self._acceptable(text, relation):
                    self.used_stems |= normalize(text, self.lang)
                    if not is_synonym:
                        self.used_synsets.add(raw.synset_id)
                    return Clue(order=order, text=text, relation=relation, synset_id=raw.synset_id)
        return None


def generate_clues(
    target_synset: str,
    lang: Lang,
    display: str,
    n: int = NUM_CLUES,
    exclude: frozenset[str] = frozenset(),
) -> list[Clue]:
    """Genera n pistas ordenadas de la más general a la más reveladora.

    Raises:
        InsufficientCluesError: si alguna posición del plan no tiene candidato válido.
    """
    if n > len(CLUE_PLAN):
        raise ValueError(f"CLUE_PLAN solo define {len(CLUE_PLAN)} posiciones")

    selector = _ClueSelector(target_synset, lang, display, exclude)
    clues: list[Clue] = []
    for order, relations in enumerate(CLUE_PLAN[:n], start=1):
        clue = selector.pick(order, relations)
        if clue is None:
            wanted = ", ".join(r.value for r in relations)
            raise InsufficientCluesError(
                f"{target_synset} [{lang.value}]: sin candidato para la pista {order} ({wanted})"
            )
        clues.append(clue)
    return clues
