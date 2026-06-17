"""Script de ejecución: entrenamiento / benchmark del modelo (MITSUI).

Lee X.parquet / y.parquet de data/processed/ (o los genera si faltan),
hace un benchmark de algoritmos para el target único (config.TARGET) y
guarda el mejor modelo en artifacts/models/.

Uso:
    python scripts/run_training.py                       # benchmark de todos
    python scripts/run_training.py --model decision_tree # un solo algoritmo
    python scripts/run_training.py --model-out artifacts/models/model.pkl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.make_dataset import build_dataset  # noqa: E402
from src.models.train_model import MODEL_NAMES, benchmark, save_model, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento / benchmark MITSUI")
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_NAMES],
        default="all",
        help="Algoritmo a entrenar; 'all' hace benchmark y guarda el mejor",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=config.MODELS_DIR / "model.pkl",
        help="Ruta de salida del modelo entrenado (.pkl)",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.2,
        help="Fracción final de la serie usada como validación temporal",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    x_path = config.PROCESSED_DATA_DIR / "X.parquet"
    y_path = config.PROCESSED_DATA_DIR / "y.parquet"
    if not (x_path.exists() and y_path.exists()):
        logging.info("Dataset procesado no encontrado; generándolo...")
        x_path, y_path = build_dataset()

    X = pd.read_parquet(x_path)
    y = pd.read_parquet(y_path)

    if args.model == "all":
        results, pipelines = benchmark(X, y, valid_fraction=args.valid_fraction)
        best = results.iloc[0]["model"]
        save_model(pipelines[best], args.model_out)
        print("\n=== Benchmark (target:", config.TARGET, "-", config.TARGET_DEFINITION, ") ===")
        print(results.to_string(index=False))
        print(f"\nMejor modelo: {best}  ->  {args.model_out}")
    else:
        model, metrics = train(X, y, model_name=args.model, valid_fraction=args.valid_fraction)
        save_model(model, args.model_out)
        print(f"OK  {args.model} -> {args.model_out}  | métricas: {metrics}")


if __name__ == "__main__":
    main()
