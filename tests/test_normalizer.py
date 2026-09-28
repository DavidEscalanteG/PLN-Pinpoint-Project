import pytest

from pinpoint.models import Lang
from pinpoint.normalizer import clean, matches_expected_form, normalize, strip_accents


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


@pytest.mark.parametrize(
    ("singular", "plural"),
    [("oso", "osos"), ("ave", "aves"), ("mes", "meses"), ("pez", "peces"), ("luz", "luces")],
)
def test_spanish_plural_equivalence_covers_short_and_orthographic_forms(singular, plural):
    assert matches_expected_form(plural, singular, Lang.ES)


@pytest.mark.parametrize(
    ("valid", "invalid"),
    [("tesis", "tesi"), ("tesis", "tesises"), ("crisis", "crisi"), ("virus", "viruses")],
)
def test_spanish_invariant_words_reject_invented_inflections(valid, invalid):
    assert not matches_expected_form(invalid, valid, Lang.ES)


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("ours", "ours"),
        ("chat", "chats"),
        ("cheval", "chevaux"),
        ("oiseau", "oiseaux"),
        ("animal", "animaux"),
        ("cheveu", "cheveux"),
    ],
)
def test_french_plural_equivalence(singular, plural):
    assert matches_expected_form(plural, singular, Lang.FR)


@pytest.mark.parametrize(
    ("singular", "valid_plural", "invalid_plural"),
    [("bleu", "bleus", "bleux"), ("pneu", "pneus", "pneux"), ("bal", "bals", "baux")],
)
def test_french_plural_exceptions(singular, valid_plural, invalid_plural):
    assert matches_expected_form(valid_plural, singular, Lang.FR)
    assert not matches_expected_form(invalid_plural, singular, Lang.FR)


@pytest.mark.parametrize(
    ("singular", "plural"),
    [("bear", "bears"), ("city", "cities"), ("knife", "knives"), ("mouse", "mice"),
     ("goose", "geese"), ("child", "children")],
)
def test_english_plural_equivalence(singular, plural):
    assert matches_expected_form(plural, singular, Lang.EN)
