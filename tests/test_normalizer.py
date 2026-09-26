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


@pytest.mark.parametrize(("a", "b"), [("perro", "los perros"), ("león", "leones"), ("canción", "canciones")])
def test_spanish_inflections_match(a, b):
    assert normalize(a, Lang.ES) == normalize(b, Lang.ES)


def test_multiword_order_independent():
    assert normalize("helado de crema", Lang.ES) == normalize("crema helado", Lang.ES)


def test_only_stopwords_is_not_empty():
    assert normalize("the", Lang.EN)
