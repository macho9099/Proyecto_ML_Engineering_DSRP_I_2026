# -*- coding: utf-8 -*-
"""Inferencia / predicción para el reto MITSUI Commodity Prediction.

Problema simplificado: un único target (`config.TARGET`).

Función reutilizable:
    predict(model, X) -> DataFrame con date_id + <config.TARGET>
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_model(path: Path | None = None):
    path = Path(path) if path else config.MODELS_DIR / "model.pkl"
    logger.info("Cargando modelo desde %s", path)
    return joblib.load(path)


def predict(model, X: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Genera predicciones para el target único.

    Returns un DataFrame con `date_id` + la columna `<config.TARGET>`.
    """
    target = target or config.TARGET
    feats = [c for c in X.columns if c != config.ID_COL]
    preds = model.predict(X[feats].to_numpy())
    out = pd.DataFrame({config.ID_COL: X[config.ID_COL].to_numpy(), target: preds})
    return out


def save_predictions(preds: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path else config.PREDICTIONS_DIR / "preds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(path, index=False)
    logger.info("Predicciones guardadas en %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.data.make_dataset import load_test, prepare_features

    model = load_model()
    test = prepare_features(load_test().drop(columns=["is_scored"], errors="ignore"))
    preds = predict(model, test)
    save_predictions(preds)
