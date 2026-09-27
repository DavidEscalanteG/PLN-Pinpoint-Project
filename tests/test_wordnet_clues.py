import pytest

from pinpoint.bank import load_categories
from pinpoint.config import FAR_HYPERNYM_MAX_LEVEL, FAR_HYPERNYM_MIN_LEVEL, NUM_CLUES
from pinpoint.language import display_answer, lemmas_in
from pinpoint.matcher import leaks_answer
from pinpoint.models import InsufficientCluesError, Lang, Relation
from pinpoint.normalizer import normalize
from pinpoint.wordnet_clues import (
    _looks_untranslated,
    candidate_pools,
    far_hypernyms,
    generate_clues,
)

GENERAL = {Relation.HYPERNYM_FAR, Relation.HYPERNYM}
BANK = load_categories()


# ---------- Candidatos (independientes del idioma) ----------

def test_pools_contain_required_relations():
    pools = candidate_pools("dog.n.01")
    assert pools[Relation.HYPERNYM]
    assert pools[Relation.HYPONYM]
    assert pools[Relation.SYNONYM]
    assert all(r.synset_id != "dog.n.01" for r in pools[Relation.HYPONYM])


def test_far_hypernyms_respect_level_range():
    levels = [level for level, _ in far_hypernyms("horse.n.01")]
    assert levels
    assert all(FAR_HYPERNYM_MIN_LEVEL <= lvl <= FAR_HYPERNYM_MAX_LEVEL for lvl in levels)


@pytest.mark.parametrize(
    ("synset", "expected_first"),
    [
        ("horse.n.01", "mammal.n.01"),   # antes: odd-toed_ungulate (tecnicismo)
        ("bird.n.01", "animal.n.01"),    # antes: chordate
        ("tree.n.01", "plant.n.02"),     # antes: vascular_plant
    ],
)
def test_far_hypernyms_prefer_known_words(synset, expected_first):
    assert far_hypernyms(synset)[0][1] == expected_first


def test_far_hypernyms_exclude_direct_parents():
    parents = {r.synset_id for r in candidate_pools("dog.n.01")[Relation.HYPERNYM]}
    assert parents.isdisjoint(name for _, name in far_hypernyms("dog.n.01"))


# ---------- Heurística de lemas sin traducir ----------

def test_untranslated_lemma_detected():
    assert _looks_untranslated("craniate", "vertebrate.n.01", Lang.ES)


@pytest.mark.parametrize(("form", "synset"), [("animal", "animal.n.01"), ("alcohol", "alcohol.n.01")])
def test_cognates_are_not_untranslated(form, synset):
    assert not _looks_untranslated(form, synset, Lang.ES)


def test_english_never_untranslated():
    assert not _looks_untranslated("craniate", "vertebrate.n.01", Lang.EN)


# ---------- Generación de pistas ----------

def test_direct_hypernym_may_extend_first_clue():
    clues = generate_clues("tree.n.01", Lang.EN, "tree")
    assert (clues[0].text, clues[1].text) == ("plant", "woody plant")
    assert clues[1].relation is Relation.HYPERNYM


def test_generation_is_deterministic():
    assert generate_clues("horse.n.01", Lang.EN, "horse") == generate_clues("horse.n.01", Lang.EN, "horse")


def test_insufficient_clues_raises():
    pools = candidate_pools("dog.n.01")
    everything = frozenset(r.synset_id for pool in pools.values() for r in pool)
    with pytest.raises(InsufficientCluesError):
        generate_clues("dog.n.01", Lang.EN, "dog", exclude=everything)


@pytest.mark.parametrize("lang", [Lang.EN, Lang.ES])
@pytest.mark.parametrize("category", BANK, ids=[c.synset_id for c in BANK])
def test_every_bank_category_yields_valid_clues(category, lang):
    """Contrato de todo el banco: 5 pistas, ordenadas, sin fugas y sin repetir raíces."""
    answer = display_answer(category, lang)
    clues = generate_clues(category.synset_id, lang, answer, exclude=category.exclude)
    answers = [answer, *lemmas_in(category.synset_id, lang)]

    assert len(clues) == NUM_CLUES
    assert [c.order for c in clues] == list(range(1, NUM_CLUES + 1))
    assert clues[0].relation in GENERAL | {Relation.SIBLING}
    assert len({c.text.lower() for c in clues}) == NUM_CLUES
    assert all(c.synset_id not in category.exclude for c in clues)

    for clue in clues:
        if clue.relation is not Relation.SYNONYM:
            # Misma regla que el generador: raíz contra todas las respuestas,
            # subcadena solo contra la canónica (evita descartar 'cánido' por el lema 'can').
            assert not leaks_answer(clue.text, answers, lang, substring_answers=[answer]), clue

    # Solo la pista 2 (hiperónimo) puede compartir raíces con la 1, y solo si la extiende.
    for i, a in enumerate(clues):
        for b in clues[i + 1:]:
            shared = normalize(a.text, lang) & normalize(b.text, lang)
            if shared:
                assert (a.order, b.order) == (1, 2) and b.relation is Relation.HYPERNYM, (a, b)
