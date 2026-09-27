"""Generación de pistas a partir de relaciones de WordNet.

Relaciones explotadas (ver config.CLUE_PLAN para el orden):
    - Hiperónimo lejano (2 a 5 niveles)    -> pista más general (1)
    - Hiperónimo directo                   -> pista general (2)
    - Hipónimos (por frecuencia de uso)    -> pistas específicas (3-4)
    - Sinónimos del mismo synset           -> pista más reveladora (5)
    - Co-hipónimos (hermanos)              -> respaldo cuando falta alguna relación
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from pinpoint.config import (
    CLUE_PLAN,
    FAR_HYPERNYM_MAX_LEVEL,
    FAR_HYPERNYM_MIN_LEVEL,
    GENERIC_SYNSETS,
    NUM_CLUES,
)
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
def far_hypernyms(synset_id: str) -> tuple[tuple[int, str], ...]:
    """Ancestros entre FAR_HYPERNYM_MIN_LEVEL y FAR_HYPERNYM_MAX_LEVEL niveles, como (nivel, synset).

    Orden de preferencia para la pista 1:
        1. Palabras conocidas (frecuencia > 0 en SemCor) antes que tecnicismos
           ('mammal' antes que 'perissodactyl').
        2. Nombres comunes antes que propios.
        3. El más cercano: lo bastante general para ser difícil sin ser inútil.
        4. Más frecuente; el nombre desempata (determinismo).
    Cada ancestro aparece una sola vez, en su nivel más cercano (búsqueda en anchura).
    """
    target = get_synset(synset_id)
    seen = {target.name()}
    frontier = [target]
    found: list[tuple[int, Synset]] = []
    for level in range(1, FAR_HYPERNYM_MAX_LEVEL + 1):
        next_frontier: list[Synset] = []
        for synset in frontier:
            for parent in _hypernyms(synset):
                if parent.name() not in seen:
                    seen.add(parent.name())
                    next_frontier.append(parent)
        if level >= FAR_HYPERNYM_MIN_LEVEL:
            found.extend((level, s) for s in next_frontier if s.name() not in GENERIC_SYNSETS)
        frontier = next_frontier

    found.sort(key=lambda item: (
        _popularity(item[1]) == 0,
        _is_proper_noun(item[1]),
        item[0],
        -_popularity(item[1]),
        item[1].name(),
    ))
    return tuple((level, s.name()) for level, s in found)


@lru_cache(maxsize=None)
def candidate_pools(synset_id: str) -> dict[Relation, tuple[RawClue, ...]]:
    """Candidatos por relación, independientes del idioma y ordenados por preferencia."""
    target = get_synset(synset_id)
    parents = _by_popularity(_hypernyms(target))
    children = _by_popularity(_unique(_hyponyms(target), {synset_id}))
    grandchildren = _by_popularity(_unique((g for c in children for g in _hyponyms(c)), {synset_id}))
    siblings = _by_popularity(_unique((s for p in parents for s in _hyponyms(p)), {synset_id}))

    def raw(synsets: Iterable[Synset], relation: Relation) -> tuple[RawClue, ...]:
        return tuple(RawClue(s.name(), relation) for s in synsets)

    return {
        Relation.HYPERNYM_FAR: tuple(
            RawClue(name, Relation.HYPERNYM_FAR) for _, name in far_hypernyms(synset_id)
        ),
        Relation.HYPERNYM: raw(_unique(parents, {synset_id}), Relation.HYPERNYM),
        # Hipónimos directos primero; los de segundo nivel amplían el banco de candidatos.
        Relation.HYPONYM: raw(children, Relation.HYPONYM)
        + raw([g for g in grandchildren if g not in children], Relation.HYPONYM),
        Relation.SIBLING: raw(siblings, Relation.SIBLING),
        Relation.SYNONYM: (RawClue(synset_id, Relation.SYNONYM),),
    }


@lru_cache(maxsize=4096)
def _looks_untranslated(form: str, synset_id: str, lang: Lang) -> bool:
    """Heurística para lemas que OMW dejó en inglés.

    La forma coincide con un lema inglés del synset y además solo existe en un
    synset del idioma. Los cognados reales ('animal', 'alcohol', 'piano') aparecen
    en varios synsets; los lemas sin traducir ('craniate', 'diapsid') en uno solo.
    """
    if lang is Lang.EN:
        return False
    english = {l.lower() for l in lemmas_in(synset_id, Lang.EN)}
    if form.lower() not in english:
        return False
    return len(wn.synsets(form.replace(" ", "_"), lang=lang.value)) <= 1


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
        self.far_stems: frozenset[str] = frozenset()

    def _acceptable(self, text: str, relation: Relation) -> bool:
        stems = normalize(text, self.lang)
        if not stems:
            return False
        overlap = stems & self.used_stems
        # El hiperónimo directo puede especializar a la pista 1 ('plant' -> 'woody plant'):
        # comparte raíces con ella pero agrega información. Cualquier otro solapamiento
        # indica una pista repetida o demasiado parecida.
        extends_far = (
            relation is Relation.HYPERNYM
            and overlap <= self.far_stems
            and stems > self.far_stems
        )
        if overlap and not extends_far:
            return False
        # Un sinónimo es, por definición, una respuesta válida: se compara contra las demás.
        answers = (
            [a for a in self.answers if a.lower() != text.lower()]
            if relation is Relation.SYNONYM
            else self.answers
        )
        return not leaks_answer(text, answers, self.lang, substring_answers=[self.display])

    def _preferred_forms(self, raw: RawClue) -> list[str]:
        """Formas de la pista en orden de preferencia.

        - Descarta variantes con la misma raíz y se queda con la más corta
          ('animal' en vez de 'animales', 'perro' en vez de 'perros').
        - En idiomas distintos al inglés, pone al final las formas que parecen
          no traducidas ('vertebrado' antes que 'craniate'); ver _looks_untranslated.
        """
        forms = surface_forms(raw, self.target, self.lang, self.display)
        by_stem: dict[frozenset[str], str] = {}
        for form in forms:
            stem = normalize(form, self.lang)
            if stem not in by_stem or len(form) < len(by_stem[stem]):
                by_stem[stem] = form
        kept = [f for f in forms if f in by_stem.values()]
        return sorted(kept, key=lambda f: _looks_untranslated(f, raw.synset_id, self.lang))

    def pick(self, order: int, relations: tuple[Relation, ...]) -> Clue | None:
        # 1.ª pasada: sin nombres propios ni lemas sin traducir. 2.ª pasada: se permiten.
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
            for text in self._preferred_forms(raw):
                # 1.ª pasada: se evitan nombres propios y palabras que OMW dejó sin traducir.
                # Los sinónimos pueden ser nombres científicos ("Canis familiaris"): se permiten siempre.
                if not allow_proper and not is_synonym and (
                    text[:1].isupper() or _looks_untranslated(text, raw.synset_id, self.lang)
                ):
                    continue
                if self._acceptable(text, relation):
                    stems = normalize(text, self.lang)
                    self.used_stems |= stems
                    if relation is Relation.HYPERNYM_FAR:
                        self.far_stems = stems
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
