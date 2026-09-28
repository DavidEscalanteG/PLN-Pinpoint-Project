
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
    #Analiza y valida los argumentos de la línea de comandos
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


def formato(pista: Clue, relacion: bool) -> str:
    #Formatea una pista para mostrarla en pantalla, incluyendo su relación segun wordnet
    sufijo = f"   [{pista.relation.value}]" if relacion else ""
    return f"  Pista {pista.order}/{NUM_CLUES}: {pista.text.upper()}{sufijo}"


def pistas_reveladas(sesion: GameSession, relacion: bool) -> None:
    #Muestra todas las pistas reveladas hasta ahora
    pistas = sesion.revealed_clues()
    if len(pistas) <= 1:
        return
    print()
    print("  Pistas reveladas hasta ahora:")
    for pista in pistas[:-1]:
        sufijo = f"   [{pista.relation.value}]" if relacion else ""
        print(f"    {pista.order}. {pista.text.upper()}{sufijo}")


def leer(pistas_restantes: int) -> str:
    #Solicita al jugador una respuesta hasta recibir un texto no vacío
    while True:
        text = input(f"  Tu respuesta ({pistas_restantes} pista(s) más disponible(s)) > ").strip()
        if text:
            return text
        print("  Escribe una respuesta, ':pasar' para ver otra pista o ':salir' para terminar.")


def jugar(sesion: GameSession, relacion: bool) -> bool:
    #Juega una ronda
    print(f"\n=== Ronda {sesion.rounds_played + 1} de {sesion.total_rounds} ===")
    while sesion.round_active:
        print()
        pistas_reveladas(sesion, relacion)
        partida = sesion.current_clue()
        print(formato(partida, relacion))
        print()
        pistas_restantes = NUM_CLUES - partida.order
        texto = leer(pistas_restantes)
        comando = texto.lower()
        if comando in QUIT_COMMANDS:
            print("  Partida interrumpida por el jugador.")
            return False

        respuesta = sesion.pass_turn() if comando in PASS_COMMANDS else sesion.guess(texto)
        if respuesta.correct:
            print(f"  ✔ ¡CORRECTO! La respuesta era '{respuesta.revealed_answer}'.")
            print(f"    Pistas usadas: {respuesta.clues_used}/{NUM_CLUES}  →  +{respuesta.points} pts")
        elif respuesta.finished:
            print(f"  ✘ Se acabaron las pistas. La respuesta correcta era: '{respuesta.revealed_answer}' (0 pts)")
        else:
            verbo = "Pista saltada" if comando in PASS_COMMANDS else f"'{texto}' no es correcto"
            print(f"  ✘ {verbo}. Revelando la siguiente pista...")

    print("\n  Resumen de la ronda — pistas y su relación en WordNet:")
    for pista in sesion.round.clues: 
        print(f"    {pista.order}. {pista.text:<28} {pista.relation.value:<24} ({pista.synset_id})")
    return True


def resultado(sesion: GameSession) -> None:
    #Resumen de los resultados de la partida
    puntos = sesion.summary
    rondas = len(puntos.results)
    print("\n" + "=" * 32)
    print("  RESUMEN DE LA PARTIDA")
    print("=" * 32)
    if rondas == 0:
        print("  No se completó ninguna ronda.")
        return
    for i, respuesta in enumerate(puntos.results, start=1):
        estado = "✔ acertada" if respuesta.correct else "✘ fallada"
        print(f"  Ronda {i}: {estado}  ({respuesta.clues_used}/{NUM_CLUES} pistas, +{respuesta.points} pts)")
    print("-" * 32)
    print(f"  Rondas jugadas: {rondas}   Acertadas: {puntos.rounds_won}")
    print(f"  Puntaje total: {puntos.total_points} / {max_points(rondas)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    lang = Lang(args.lang)

    try:
        sesion = GameSession(lang, rounds=args.rounds, seed=args.seed)
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
        while not sesion.is_over:
            try:
                sesion.new_round()
            except CategoryBankError as exc:
                print(f"\n{exc}")
                break
            if not jugar(sesion, args.show_relations):
                break
    except (KeyboardInterrupt, EOFError):
        print("\n  Partida interrumpida.")

    resultado(sesion)
    return 0


if __name__ == "__main__":
    exit_code = main()
    try:
        input("\nPresiona Enter para salir...")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(exit_code)
