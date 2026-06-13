"""Script de ejecución: preprocesamiento de datos (MITSUI Commodity Prediction).

Carga train.csv / train_labels.csv desde RAW_DATA_DIR (ver .env), aplica el
preprocesamiento y guarda X.parquet / y.parquet en data/processed/.

Uso:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --output-dir data/processed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permitir `from src import ...` aunque el paquete no esté instalado
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.make_dataset import build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesamiento MITSUI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Directorio de salida para X.parquet / y.parquet",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    x_path, y_path = build_dataset(args.output_dir)
    print(f"OK  X -> {x_path}\n    y -> {y_path}")


if __name__ == "__main__":
    main()
