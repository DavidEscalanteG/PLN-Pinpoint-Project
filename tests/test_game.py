import json

import pytest

from pinpoint.bank import load_categories
from pinpoint.game import GameSession, build_round
from pinpoint.models import Category, CategoryBankError, Lang, PinpointError, Relation

DOG = Category("dog.n.01", {"spa": "perro"})
CAT = Category("cat.n.01", {"spa": "gato"})
BEAR = Category("bear.n.01", {"spa": "oso"})


def test_build_round_excludes_shown_synonyms_from_answers():
    rnd = build_round(DOG, Lang.EN)
    shown = {c.text.lower() for c in rnd.clues if c.relation is Relation.SYNONYM}
    assert rnd.display_answer == "dog"
    assert not shown & {a.lower() for a in rnd.valid_answers}


def test_correct_on_first_clue_gives_max_points():
    session = GameSession(Lang.ES, rounds=1, categories=[DOG])
    session.new_round()
    result = session.guess("Los perros")
    assert result.correct and result.finished
    assert result.clues_used == 1 and result.points == 5
    assert session.is_over


def test_short_spanish_plural_is_accepted_in_a_real_round():
    session = GameSession(Lang.ES, rounds=1, categories=[BEAR])
    session.new_round()
    result = session.guess("Osos")
    assert result.correct and result.finished
    assert result.revealed_answer == "oso"


def test_wrong_answers_reveal_all_clues_then_answer():
    session = GameSession(Lang.ES, rounds=1, categories=[DOG])
    session.new_round()
    for expected_clue in range(1, 5):
        assert session.current_clue().order == expected_clue
        assert not session.guess("jirafa").finished
    final = session.guess("jirafa")
    assert final.finished and not final.correct
    assert final.points == 0 and final.revealed_answer == "perro"


def test_pass_turn_advances():
    session = GameSession(Lang.EN, rounds=1, categories=[DOG])
    session.new_round()
    session.pass_turn()
    assert session.current_clue().order == 2


def test_empty_guess_rejected():
    session = GameSession(Lang.EN, rounds=1, categories=[DOG])
    session.new_round()
    with pytest.raises(ValueError):
        session.guess("   ")


def test_cannot_guess_without_round():
    session = GameSession(Lang.EN, rounds=1, categories=[DOG])
    with pytest.raises(PinpointError):
        session.guess("dog")


def test_rounds_are_capped_and_not_repeated():
    session = GameSession(Lang.EN, rounds=10, seed=3, categories=[DOG, CAT])
    assert session.total_rounds == 2
    seen = set()
    while not session.is_over:
        seen.add(session.new_round().target_synset)
        session.pass_turn(), session.pass_turn(), session.pass_turn(), session.pass_turn(), session.pass_turn()
    assert seen == {"dog.n.01", "cat.n.01"}


def test_invalid_synset_in_bank_raises():
    session = GameSession(Lang.EN, rounds=1, categories=[Category("nonexistent.n.99")])
    with pytest.raises(CategoryBankError):
        session.new_round()


def test_load_categories_validates_structure(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"categories": [{"display": {}}]}), encoding="utf-8")
    with pytest.raises(CategoryBankError):
        load_categories(bad)


def test_default_bank_loads():
    assert len(load_categories()) >= 30
