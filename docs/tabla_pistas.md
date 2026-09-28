# Pista → relación de WordNet

Generado con `generate_clues()` (sin semilla: el generador es determinista).
Las pistas van de la más general (1) a la más reveladora (5).

## Caballo — `horse.n.01`

| # | Inglés | Español | Relación | Synset |
|---|--------|---------|----------|--------|
| 1 | mammal | mamífero | Hiperónimo lejano (2–5 niveles) | `mammal.n.01` |
| 2 | equine | équido | Hiperónimo directo | `equine.n.01` |
| 3 | mount | cabalgadura | Hipónimo | `saddle_horse.n.01` |
| 4 | mare | poney | Hipónimo | `mare.n.01` / `pony.n.01` |
| 5 | pony | equinos | Hipónimo / Sinónimo (mismo synset) | `pony.n.01` / `horse.n.01` |

## Árbol — `tree.n.01`

| # | Inglés | Español | Relación | Synset |
|---|--------|---------|----------|--------|
| 1 | plant | flora | Hiperónimo lejano | `plant.n.02` |
| 2 | woody plant | planta leñosa | Hiperónimo directo | `woody_plant.n.01` |
| 3 | willow | sauce | Hipónimo | `willow.n.01` |
| 4 | beech | fagus | Hipónimo | `beech.n.01` |
| 5 | ash | fresno | Hipónimo | `ash.n.02` |

## Notas para la exposición

- **Mismo synset, distinto idioma:** la pista 1 de *árbol* sale de `plant.n.02` en
  los dos idiomas. OMW solo cambia la palabra que se muestra (*plant* / *flora*).
- **La pista 2 puede extender a la 1:** *plant → woody plant*. Es la única vez que
  dos pistas pueden compartir raíz.
- **Filtro de fugas (caballo):** en la pista 4, el inglés usa *mare* y el español
  usa *poney*. La traducción de `mare.n.01` es *yegua*, pero OMW también la pone
  como lema de `horse.n.01`, así que *yegua* cuenta como respuesta válida y se
  descarta.
- **Filtro de fugas (perro) y respaldo por co-hipónimo:** en español, los dos
  hiperónimos directos fallan:
  - *cánido* (`canine.n.02`): Snowball lo reduce a la raíz `can`, que es igual al
    lema *can* de perro.
  - *animal doméstico* (`domestic_animal.n.01`): repite la raíz de la pista 1
    (*animal*).

  La pista 2 queda como *callejero* (`stray.n.01`, que es hermano de `dog.n.01`).
- **Sin sinónimo en la pista 5:** *tree* no tiene otro lema en su synset, así que
  la pista 5 es otro hipónimo (ver `CLUE_PLAN`). En *horse* pasa lo mismo por
  curaduría: el banco excluye *Equus caballus*, y la pista 5 queda como *pony*.
