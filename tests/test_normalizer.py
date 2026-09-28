import pytest

from pinpoint.models import Lang
from pinpoint.normalizer import clean, normalize, strip_accents


def test_strip_accents():
    assert strip_accents("canción ñandú Éléphant") == "cancion nandu Elephant"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ¡El PERRO!  ", "el perro"),
        ("ice_cream", "ice cream"),
        ("self-propelled   vehicle", "self propelled vehicle"),
        ("¿Qué es?", "que es"),
    ],
)
def test_clean(raw, expected):
    assert clean(raw) == expected


def test_clean_rejects_non_string():
    with pytest.raises(TypeError):
        clean(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(("a", "b"), [("dog", "dogs"), ("goose", "geese"), ("a mouse", "the mice")])
def test_english_inflections_match(a, b):
    assert normalize(a, Lang.EN) == normalize(b, Lang.EN)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("perro", "los perros"),
        ("león", "leones"),
        ("canción", "canciones"),
        ("gato", "gata"),
        ("médico", "médica"),
    ],
)
def test_spanish_inflections_match(a, b):
    assert normalize(a, Lang.ES) == normalize(b, Lang.ES)


@pytest.mark.parametrize(
    ("base", "diminutive"),
    [("perro", "perrito"), ("perro", "perritos"), ("gato", "gatita"), ("flor", "florecita")],
)
def test_spanish_diminutives_match(base, diminutive):
    assert normalize(base, Lang.ES) == normalize(diminutive, Lang.ES)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("helado de crema", "crema helado"),
        ("LOS PERROS DOMÉSTICOS", "perro domestico"),
        ("¡Máquina de escribir!", "maquina escribir"),
    ],
)
def test_multiword_answers_ignore_order_articles_accents_and_case(a, b):
    assert normalize(a, Lang.ES) == normalize(b, Lang.ES)


def test_only_stopwords_is_not_empty():
    assert normalize("the", Lang.EN)
