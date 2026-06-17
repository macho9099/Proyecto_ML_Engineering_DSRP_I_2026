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
from src.models.train_model import (  # noqa: E402
    MODEL_NAMES,
    benchmark,
    cross_validate_model,
    fit_full,
    save_model,
)


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
        "--n-splits",
        type=int,
        default=5,
        help="Nº de folds para la validación cruzada temporal (TimeSeriesSplit)",
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
        results, pipelines = benchmark(X, y, n_splits=args.n_splits)
        best = results.iloc[0]["model"]
        save_model(pipelines[best], args.model_out)
        print("\n=== Benchmark CV temporal (target:", config.TARGET,
              "-", config.TARGET_DEFINITION, f"| {args.n_splits} folds) ===")
        print(results.to_string(index=False))
        print(f"\nMejor modelo: {best}  ->  {args.model_out}")
    else:
        agg, folds = cross_validate_model(X, y, model_name=args.model, n_splits=args.n_splits)
        model = fit_full(X, y, model_name=args.model)
        save_model(model, args.model_out)
        print(f"\n=== CV temporal: {args.model} ({args.n_splits} folds) ===")
        print(folds.to_string(index=False))
        print(f"\nrmse_mean={agg['rmse_mean']:.5f}±{agg['rmse_std']:.5f}  "
              f"r2_mean={agg['r2_mean']:.4f}  corr_mean={agg['corr_mean']:.4f}")
        print(f"Modelo final (todos los datos) -> {args.model_out}")


if __name__ == "__main__":
    main()
