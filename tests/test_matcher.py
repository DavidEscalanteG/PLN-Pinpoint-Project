import pytest

from pinpoint.matcher import is_correct, leaks_answer
from pinpoint.models import Lang


@pytest.mark.parametrize("guess", ["perro", "Perros", "el perro", "PERRÓ", "perrro"])
def test_spanish_correct(guess):
    assert is_correct(guess, ["perro", "can"], Lang.ES)


@pytest.mark.parametrize("guess", ["gato", "", "   ", "perico"])
def test_spanish_incorrect(guess):
    assert not is_correct(guess, ["perro"], Lang.ES)


def test_english_multiword():
    assert is_correct("Domestic Dogs", ["dog", "domestic dog"], Lang.EN)


def test_short_words_have_no_typo_tolerance():
    assert not is_correct("car", ["cat"], Lang.EN)


@pytest.mark.parametrize("clue", ["hunting dog", "hotdog", "dogs"])
def test_leaks_detected(clue):
    assert leaks_answer(clue, ["dog"], Lang.EN)


def test_no_leak_for_unrelated_clue():
    assert not leaks_answer("canine", ["dog"], Lang.EN)


def test_substring_check_can_be_limited():
    # 'cánido' contiene 'can' (lema de dog.n.01) pero no la respuesta canónica 'perro'.
    assert not leaks_answer("zorro", ["perro", "can"], Lang.ES, substring_answers=["perro"])
