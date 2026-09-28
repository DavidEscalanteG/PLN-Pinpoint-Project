# Pinpoint WordNet

Juego de adivinanza de categorías estilo *Pinpoint* (LinkedIn). Las 5 pistas de cada ronda se generan automáticamente a partir de las relaciones léxico-semánticas de WordNet, y las respuestas del jugador se validan con lematización/stemming.

Proyecto 1 · Procesamiento de Lenguaje Natural.

## Instalación y ejecución

Requisitos: Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_nltk.py               # descarga wordnet, omw y stopwords (una sola vez)

python cli.py                      # español, 3 rondas
python cli.py --lang eng --rounds 5
python cli.py --lang spa --seed 42 --show-relations   # partida reproducible, muestra la relación de cada pista
```

Comandos dentro del juego: `:pasar` revela la siguiente pista (cuenta como intento fallido) y `:salir` termina la partida.

Pruebas y utilidades:

```bash
python -m pytest -q
python scripts/validate_bank.py --langs eng spa --show   # revisa las pistas de todo el banco
python scripts/validate_bank.py --try owl.n.01 --show    # prueba un synset nuevo antes de agregarlo
```

## Mecánica

1. El sistema toma al azar una categoría del banco (`data/categories.json`) sin repetirla dentro de la partida.
2. Genera 5 pistas desde WordNet, ordenadas de la más general a la más reveladora.
3. Muestra una pista y espera la respuesta del jugador:
   - Si acierta, la ronda termina y se asignan puntos.
   - Si falla, se revela la siguiente pista.
   - Si falla la pista 5, se muestra la respuesta correcta.
4. Al terminar cada ronda se muestran todas las pistas junto con la relación de WordNet de la que provienen.

**Puntuación:** acertar en la pista *k* da `6 − k` puntos (5, 4, 3, 2, 1). Fallar la ronda da 0. El máximo por partida es `5 × rondas`.

## Uso de WordNet (`pinpoint/wordnet_clues.py`)

Cada categoría es un **synset** (por ejemplo `dog.n.01`), no una palabra suelta, para evitar la ambigüedad de sentidos. Las pistas se generan desde `nltk.corpus.wordnet`:

| Pista | Relación preferida | Respaldo si no hay candidato válido | Nivel |
|---|---|---|---|
| 1 | Hiperónimo a 2 niveles (`hypernyms()` del hiperónimo) | hiperónimo, co-hipónimo | Muy general |
| 2 | Hiperónimo directo (`hypernyms()`, `instance_hypernyms()`) | co-hipónimo, hipónimo | General |
| 3 | Hipónimo (`hyponyms()`, `instance_hyponyms()`) | co-hipónimo | Específica |
| 4 | Hipónimo | co-hipónimo, sinónimo | Específica |
| 5 | Sinónimo, es decir, otro lema del mismo synset (`lemma_names()`) | hipónimo, co-hipónimo | Reveladora |

Los co-hipónimos son los "hermanos": otros hipónimos del mismo hiperónimo. Solo se usan como respaldo, y la CLI indica siempre de qué relación proviene cada pista.

Ejemplo con `dog.n.01` en inglés:

| # | Pista | Relación |
|---|---|---|
| 1 | animal | hiperónimo (2 niveles) |
| 2 | canine | hiperónimo |
| 3 | puppy | hipónimo |
| 4 | cur | hipónimo |
| 5 | Canis familiaris | sinónimo |

### Filtros de calidad

- **Sin fugas de la respuesta.** Se descarta cualquier pista que comparta raíz con alguna respuesta válida (*hunting dog* → *dog*) o que contenga la respuesta como subcadena (*hotdog*). Ver `matcher.leaks_answer`.
- **Sin pistas redundantes.** No se repiten synsets ni raíces entre las pistas de una misma ronda.
- **Preferencia por palabras conocidas.** Los candidatos se ordenan por frecuencia en SemCor (`lemma.count()`). Los nombres propios, como las variedades de manzana, y los lemas que OMW dejó sin traducir solo se usan si no hay otra opción.
- **Hiperónimos demasiado abstractos excluidos.** Por ejemplo `entity`, `object` o `whole` (`config.GENERIC_SYNSETS`).
- **Curaduría manual.** Cada categoría del banco puede definir `exclude`, una lista de synsets que se vetan porque generan ruido.
- **Categorías no jugables.** Si una categoría no alcanza 5 pistas válidas en el idioma elegido, se omite automáticamente y se registra un aviso.

## Normalización y comparación de respuestas (`pinpoint/normalizer.py`, `pinpoint/matcher.py`)

Se aplica el mismo pipeline a la respuesta del jugador y a cada respuesta válida:

1. **Funciones de cadenas:** minúsculas, eliminación de acentos (`unicodedata`, NFKD), `_` y `-` convertidos en espacios, eliminación de puntuación y espacios repetidos.
2. **Tokenización** por espacios.
3. **Eliminación de stopwords** de NLTK. Así, "el perro" y "a dog" equivalen a "perro" y "dog".
4. **Normalización morfológica:**
   - **Inglés:** `WordNetLemmatizer` (resuelve irregulares: *geese* → *goose*, *mice* → *mouse*) seguido de `PorterStemmer` (unifica derivaciones).
   - **Español:** reducción acotada de diminutivos frecuentes seguida de `SnowballStemmer`. El stemming resuelve plurales y flexión (*leones* → *leon*, *perros* → *perr*).
   - **Francés:** `SnowballStemmer`, ya que NLTK no incluye un lematizador para este idioma.
5. **Comparación como conjuntos de raíces.** Esto soporta respuestas de varias palabras sin importar el orden.
6. **Tolerancia a un error tipográfico** (distancia de edición ≤ 1) en respuestas de 6 caracteres o más.

Ejemplos del pipeline:

| Entrada del jugador | Forma esperada | Resultado |
|---|---|---|
| `LOS PERROS DOMÉSTICOS` | `perro domestico` | Correcta: se eliminan artículo, mayúsculas, acento y plural |
| `gata` | `gato` | Correcta: Snowball unifica la flexión de género |
| `perrito` | `perro` | Correcta: se reduce un sufijo diminutivo frecuente antes del stemming |
| `jirrafa` | `jirafa` | Correcta: se tolera una errata en palabras de al menos 6 caracteres |
| `barco` | `banco` | Incorrecta: no se aplica tolerancia tipográfica a palabras cortas |

Se evaluó usar el modelo `es_core_news_sm` de spaCy como lematizador español. No se integró porque añadiría una dependencia y un modelo externos considerablemente más pesados para la instalación y la demo. Snowball, junto con la reducción acotada de diminutivos y los casos de prueba anteriores, cubre las variantes requeridas manteniendo el proyecto reproducible con NLTK.

**Respuestas válidas:** todos los lemas del synset en el idioma de juego, excepto los sinónimos que ya se mostraron como pista (escribir la pista no cuenta como adivinar).

## Idiomas (`pinpoint/language.py`)

Las traducciones provienen del Open Multilingual WordNet (`lemma_names(lang='spa' | 'fra')`).

| Modalidad | Pistas en | Respuesta esperada en |
|---|---|---|
| `--lang spa` (por defecto) | Español | Español |
| `--lang eng` | Inglés | Inglés |
| `--lang fra` (valor agregado) | Francés | Francés |

El banco define la forma canónica de la respuesta por idioma (`display`), porque OMW entrega los lemas en orden alfabético. Por ejemplo, para `dog.n.01` el primer lema sería *can* y no *perro*.

## Estructura

```
pinpoint-wordnet/
├── cli.py                    # interfaz de consola
├── setup_nltk.py             # descarga de recursos NLTK
├── data/categories.json      # banco de categorías (synset, display, exclude)
├── pinpoint/
│   ├── models.py             # contratos de datos y excepciones
│   ├── config.py             # constantes: plan de pistas, puntuación, tolerancias
│   ├── wordnet_clues.py      # generación de pistas desde WordNet
│   ├── language.py           # capa bilingüe (OMW)
│   ├── normalizer.py         # funciones de cadenas + lematización/stemming
│   ├── matcher.py            # comparación y detección de fugas
│   ├── bank.py               # carga y validación del banco
│   ├── game.py               # lógica del juego (sin I/O)
│   └── scoring.py            # puntuación
├── scripts/validate_bank.py  # validación y curaduría del banco
└── tests/                    # pytest
```

`game.GameSession` no hace entrada ni salida. La CLI, un notebook o una web la usan de la misma forma:

```python
from pinpoint import GameSession, Lang

session = GameSession(Lang.ES, rounds=3, seed=42)
session.new_round()
print(session.current_clue().text)
result = session.guess("perro")   # GuessResult(correct, clues_used, finished, points, revealed_answer)
```

## Equipo

| Integrante | Responsabilidad | Archivos |
|---|---|---|
| P1 | Generación de pistas con WordNet | `wordnet_clues.py`, `tests/test_wordnet_clues.py` |
| P2 | Normalización y comparación | `normalizer.py`, `matcher.py`, tests |
| P3 | Lógica del juego e interfaz | `game.py`, `scoring.py`, `cli.py`, `tests/test_game.py` |
| P4 | Bilingüe, banco de categorías y documentación | `language.py`, `bank.py`, `data/`, `scripts/`, README |

## Limitaciones conocidas

- La cobertura de OMW en español y francés es incompleta y a veces ruidosa: algunos lemas quedan sin traducir o tienen sentidos poco comunes. Esto se mitiga con los filtros descritos arriba, con `exclude` y con `scripts/validate_bank.py`.
- El stemming puede unir palabras no relacionadas (*cánido* y *can* comparten la raíz *can*). En la validación de respuestas esto rara vez importa; en la generación de pistas solo provoca que se descarte algún candidato.
