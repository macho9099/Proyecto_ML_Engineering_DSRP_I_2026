"""Script de ejecución: predicción / inferencia (MITSUI Commodity Prediction).

Carga el modelo de artifacts/models/, preprocesa test.csv y guarda las
predicciones (date_id + target_0..target_423) en artifacts/predictions/.

Uso:
    python scripts/run_prediction.py
    python scripts/run_prediction.py --model artifacts/models/model.pkl \
        --output artifacts/predictions/preds.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.make_dataset import load_test, preprocess  # noqa: E402
from src.models.predict_model import load_model, predict, save_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción MITSUI")
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODELS_DIR / "model.pkl",
        help="Ruta al modelo entrenado (.pkl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PREDICTIONS_DIR / "preds.csv",
        help="Ruta de salida de las predicciones (.csv)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    model = load_model(args.model)
    test = load_test().drop(columns=["is_scored"], errors="ignore")
    test = preprocess(test)
    preds = predict(model, test)
    save_predictions(preds, args.output)
    print(f"OK  predicciones -> {args.output}  | shape={preds.shape}")


if __name__ == "__main__":
    main()
