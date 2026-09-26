"""Interfaz de línea de comandos del juego.

Uso:
    python cli.py                      # español, 3 rondas
    python cli.py --lang eng --rounds 5
    python cli.py --lang spa --seed 42 --show-relations
"""
from __future__ import annotations

import argparse
import logging
import sys

from pinpoint.config import DEFAULT_ROUNDS, NUM_CLUES
from pinpoint.game import GameSession
from pinpoint.models import CategoryBankError, Clue, Lang, PinpointError
from pinpoint.scoring import max_points

PASS_COMMANDS = {":pasar", ":p", ":pass"}
QUIT_COMMANDS = {":salir", ":q", ":quit"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinpoint WordNet: adivina la categoría con 5 pistas.")
    parser.add_argument("--lang", choices=[l.value for l in Lang], default=Lang.ES.value,
                        help="idioma de pistas y respuestas (eng, spa, fra). Default: spa")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="rondas por partida")
    parser.add_argument("--seed", type=int, default=None, help="semilla para reproducir una partida")
    parser.add_argument("--show-relations", action="store_true",
                        help="muestra la relación de WordNet de cada pista al revelarla")
    parser.add_argument("--debug", action="store_true", help="logs detallados")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds debe ser >= 1")
    return args


def format_clue(clue: Clue, show_relation: bool) -> str:
    suffix = f"   [{clue.relation.value}]" if show_relation else ""
    return f"  Pista {clue.order}/{NUM_CLUES}: {clue.text.upper()}{suffix}"


def read_answer() -> str:
    """Pide una respuesta no vacía; lanza EOFError/KeyboardInterrupt si el usuario sale."""
    while True:
        text = input("  Tu respuesta > ").strip()
        if text:
            return text
        print("  Escribe una respuesta, ':pasar' para ver otra pista o ':salir' para terminar.")


def play_round(session: GameSession, show_relations: bool) -> bool:
    """Juega una ronda. Devuelve False si el jugador pidió salir."""
    print(f"\n=== Ronda {session.rounds_played + 1} de {session.total_rounds} ===")
    while session.round_active:
        print(format_clue(session.current_clue(), show_relations))
        answer = read_answer()
        command = answer.lower()
        if command in QUIT_COMMANDS:
            return False

        result = session.pass_turn() if command in PASS_COMMANDS else session.guess(answer)
        if result.correct:
            print(f"  ✔ ¡Correcto! Era '{result.revealed_answer}'. "
                  f"Pistas usadas: {result.clues_used}. +{result.points} pts")
        elif result.finished:
            print(f"  ✘ Sin pistas. La respuesta era: '{result.revealed_answer}'")
        else:
            print("  ✘ Incorrecto.")

    print("  Pistas de la ronda y su relación en WordNet:")
    for clue in session.round.clues:  # type: ignore[union-attr]
        print(f"    {clue.order}. {clue.text:<28} {clue.relation.value:<24} ({clue.synset_id})")
    return True


def print_summary(session: GameSession) -> None:
    summary = session.summary
    played = len(summary.results)
    print("\n=== Resumen de la partida ===")
    print(f"  Rondas jugadas: {played}   Acertadas: {summary.rounds_won}")
    print(f"  Puntaje: {summary.total_points} / {max_points(played)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    lang = Lang(args.lang)

    try:
        session = GameSession(lang, rounds=args.rounds, seed=args.seed)
    except LookupError:
        print("Faltan recursos de NLTK. Ejecuta: python setup_nltk.py")
        return 1
    except PinpointError as exc:
        print(f"Error: {exc}")
        return 1

    print("PINPOINT WORDNET")
    print(f"  Pistas en: {lang.label}  |  Responde en: {lang.label}")
    print(f"  {NUM_CLUES} pistas por ronda, de la más general a la más específica.")
    print(f"  Puntos: pista 1 = {NUM_CLUES} pts ... pista {NUM_CLUES} = 1 pt.")
    print("  Comandos: ':pasar' (siguiente pista), ':salir' (terminar).")

    try:
        while not session.is_over:
            try:
                session.new_round()
            except CategoryBankError as exc:
                print(f"\n{exc}")
                break
            if not play_round(session, args.show_relations):
                break
    except (KeyboardInterrupt, EOFError):
        print("\n  Partida interrumpida.")

    print_summary(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
