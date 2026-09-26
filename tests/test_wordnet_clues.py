import pytest

from pinpoint.config import NUM_CLUES
from pinpoint.language import lemmas_in
from pinpoint.matcher import leaks_answer
from pinpoint.models import InsufficientCluesError, Lang, Relation
from pinpoint.wordnet_clues import candidate_pools, generate_clues

GENERAL = {Relation.HYPERNYM_L2, Relation.HYPERNYM}


def test_pools_contain_required_relations():
    pools = candidate_pools("dog.n.01")
    assert pools[Relation.HYPERNYM]
    assert pools[Relation.HYPONYM]
    assert pools[Relation.SYNONYM]
    assert all(r.synset_id != "dog.n.01" for r in pools[Relation.HYPONYM])


@pytest.mark.parametrize("lang", [Lang.EN, Lang.ES])
@pytest.mark.parametrize(("synset", "display"), [("dog.n.01", {"eng": "dog", "spa": "perro"}),
                                                 ("cat.n.01", {"eng": "cat", "spa": "gato"})])
def test_generates_five_ordered_clues_without_leaks(synset, display, lang):
    answer = display[lang.value]
    clues = generate_clues(synset, lang, answer)

    assert len(clues) == NUM_CLUES
    assert [c.order for c in clues] == list(range(1, NUM_CLUES + 1))
    assert clues[0].relation in GENERAL  # la primera pista es la más general
    assert len({c.text.lower() for c in clues}) == NUM_CLUES
    for clue in clues:
        if clue.relation is not Relation.SYNONYM:
            assert not leaks_answer(clue.text, [answer, *lemmas_in(synset, lang)], lang)


def test_generation_is_deterministic():
    first = generate_clues("horse.n.01", Lang.EN, "horse")
    second = generate_clues("horse.n.01", Lang.EN, "horse")
    assert first == second


def test_insufficient_clues_raises():
    # Excluir todo lo posible fuerza el error.
    pools = candidate_pools("dog.n.01")
    everything = frozenset(r.synset_id for pool in pools.values() for r in pool)
    with pytest.raises(InsufficientCluesError):
        generate_clues("dog.n.01", Lang.EN, "dog", exclude=everything)
