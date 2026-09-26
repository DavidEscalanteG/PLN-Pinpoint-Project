"""Valida que cada categoría del banco genere 5 pistas en los idiomas indicados.

Uso:
    python scripts/validate_bank.py                        # valida eng y spa
    python scripts/validate_bank.py --langs eng spa fra --show
    python scripts/validate_bank.py --prune                # reescribe el banco solo con las válidas
    python scripts/validate_bank.py --try horse.n.01 owl.n.01 --show   # prueba synsets nuevos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinpoint.bank import load_categories, save_categories  # noqa: E402
from pinpoint.config import CATEGORIES_PATH  # noqa: E402
from pinpoint.game import build_round  # noqa: E402
from pinpoint.language import lemmas_in  # noqa: E402
from pinpoint.models import Category, Lang, PinpointError  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=CATEGORIES_PATH)
    parser.add_argument("--langs", nargs="+", default=["eng", "spa"], choices=[l.value for l in Lang])
    parser.add_argument("--show", action="store_true", help="muestra las pistas generadas")
    parser.add_argument("--prune", action="store_true", help="elimina del banco las categorías inválidas")
    parser.add_argument("--try", dest="try_synsets", nargs="+", metavar="SYNSET",
                        help="valida synsets que no están en el banco")
    return parser.parse_args(argv)


def check(category: Category, langs: list[Lang], show: bool) -> bool:
    ok = True
    for lang in langs:
        try:
            rnd = build_round(category, lang)
        except PinpointError as exc:
            print(f"  [FAIL] {lang.value}: {exc}")
            ok = False
            continue

        override = category.display.get(lang.value)
        lemmas = {l.lower() for l in lemmas_in(category.synset_id, lang)}
        warn = f"  (aviso: '{override}' no es lema de WordNet)" if override and override.lower() not in lemmas else ""
        print(f"  [OK]   {lang.value}: {rnd.display_answer}{warn}")
        if show:
            for clue in rnd.clues:
                print(f"           {clue.order}. {clue.text:<28} {clue.relation.value:<24} {clue.synset_id}")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    langs = [Lang(code) for code in args.langs]

    if args.try_synsets:
        categories = [Category(s) for s in args.try_synsets]
    else:
        try:
            categories = load_categories(args.path)
        except PinpointError as exc:
            print(f"Error: {exc}")
            return 1

    valid: list[Category] = []
    for category in categories:
        print(category.synset_id)
        if check(category, langs, args.show):
            valid.append(category)

    failed = len(categories) - len(valid)
    print(f"\nVálidas: {len(valid)}/{len(categories)}  |  Fallidas: {failed}")

    if args.prune and not args.try_synsets and failed:
        save_categories(valid, args.path)
        print(f"Banco reescrito en {args.path}")
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
