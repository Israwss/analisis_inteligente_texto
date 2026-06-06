#!/usr/bin/env python
"""
Detección Inteligente de Noticias Falsas en Español
===================================================
Orquestador de línea de comandos del proyecto.

Uso:
    python main.py download                 # descarga el corpus
    python main.py stats                    # estadísticas del dataset
    python main.py classic                  # entrena los 5 modelos clásicos
    python main.py classic --analyzer char  # variante con n-gramas de carácter
    python main.py ablation                 # estudio de ablación (20 experimentos)
    python main.py beto --epochs 4          # fine-tuning de BETO (requiere GPU)
    python main.py all                      # download + ablation + clásico

Repositorio del dataset: https://github.com/jpposadas/FakeNewsCorpusSpanish (CC-BY-4.0)
"""
from __future__ import annotations

import argparse
import sys

# La consola de Windows usa cp1252 por defecto y no puede imprimir algunos
# caracteres (✓, •, ±). Forzamos UTF-8 para evitar UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def cmd_download(args):
    from src import data
    data.download(force=args.force)


def cmd_stats(args):
    from src import data
    data.download(force=False)
    print("\n== Estadísticas del corpus ==")
    for name, frame in data.load_all().items():
        n_fake = int(frame["label"].sum())
        n_true = len(frame) - n_fake
        chars = frame["text"].str.len().mean()
        print(f"{name:12s}: {len(frame):4d} notas | fake={n_fake} true={n_true} "
              f"| long. media={chars:.0f} chars")


def cmd_classic(args):
    from src import data, train_classic
    data.download(force=False)
    train_classic.run(
        analyzer=args.analyzer,
        use_stopwords=not args.no_stopwords,
        remove_accents=args.remove_accents,
        cv_folds=args.cv,
    )


def cmd_ablation(args):
    from src import data, train_classic
    data.download(force=False)
    train_classic.run_ablation(cv_folds=args.cv)


def cmd_beto(args):
    from src import data, train_beto
    data.download(force=False)
    train_beto.run(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def cmd_all(args):
    from src import data, train_classic
    data.download(force=False)
    train_classic.run_ablation(cv_folds=args.cv)
    train_classic.run(analyzer="word", cv_folds=args.cv)
    print("\nPara la comparación con transformer ejecuta:  python main.py beto")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("download", help="Descarga el corpus")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("stats", help="Estadísticas del dataset")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("classic", help="Entrena los modelos clásicos")
    sp.add_argument("--analyzer", choices=["word", "char"], default="word")
    sp.add_argument("--no-stopwords", action="store_true")
    sp.add_argument("--remove-accents", action="store_true")
    sp.add_argument("--cv", type=int, default=5)
    sp.set_defaults(func=cmd_classic)

    sp = sub.add_parser("ablation", help="Estudio de ablación (20 experimentos)")
    sp.add_argument("--cv", type=int, default=5)
    sp.set_defaults(func=cmd_ablation)

    sp = sub.add_parser("beto", help="Fine-tuning de BETO (GPU)")
    sp.add_argument("--epochs", type=int, default=4)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.set_defaults(func=cmd_beto)

    sp = sub.add_parser("all", help="download + ablación + clásico")
    sp.add_argument("--cv", type=int, default=5)
    sp.set_defaults(func=cmd_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
