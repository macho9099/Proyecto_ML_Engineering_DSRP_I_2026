"""Script de ejecución: entrenamiento del modelo (MITSUI Commodity Prediction).

Lee X.parquet / y.parquet de data/processed/ (o los genera si faltan),
entrena el regresor multi-objetivo y guarda el modelo en artifacts/models/.

Uso:
    python scripts/run_training.py
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
from src.models.train_model import save_model, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento MITSUI")
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

    model, metrics = train(X, y, valid_fraction=args.valid_fraction)
    save_model(model, args.model_out)
    print(f"OK  modelo -> {args.model_out}  | métricas: {metrics}")


if __name__ == "__main__":
    main()
