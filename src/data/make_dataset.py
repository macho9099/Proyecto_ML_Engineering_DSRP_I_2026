# -*- coding: utf-8 -*-
"""Carga y preprocesamiento de los datos del reto MITSUI Commodity Prediction.

Funciones reutilizables (importables desde notebooks y scripts):
    load_raw()        -> (features, labels, target_pairs)
    preprocess()      -> features listas para modelar
    build_dataset()   -> guarda X/y procesados en data/processed/
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSV crudos de la competencia.

    Returns
    -------
    features : DataFrame  (date_id + ~557 columnas de precios)
    labels   : DataFrame  (date_id + target_0..target_423)
    pairs    : DataFrame  (target, lag, pair)
    """
    logger.info("Cargando datos crudos desde %s", config.RAW_DATA_DIR)
    features = pd.read_csv(config.TRAIN_CSV)
    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    pairs = pd.read_csv(config.TARGET_PAIRS_CSV)
    logger.info(
        "features=%s  labels=%s  pairs=%s", features.shape, labels.shape, pairs.shape
    )
    return features, labels, pairs


def load_test() -> pd.DataFrame:
    """Carga test.csv (incluye la columna `is_scored`)."""
    return pd.read_csv(config.TEST_CSV)


def preprocess(features: pd.DataFrame) -> pd.DataFrame:
    """Preprocesamiento base de las features.

    Las series financieras tienen valores faltantes (instrumentos que no
    cotizan ciertos días). Estrategia simple y robusta para series de tiempo:
    forward-fill y luego back-fill, ordenando por `date_id`.
    """
    df = features.sort_values(config.ID_COL).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != config.ID_COL]
    df[feature_cols] = df[feature_cols].ffill().bfill()
    return df


def build_dataset(output_dir: Path | None = None) -> tuple[Path, Path]:
    """Genera y persiste el dataset procesado (X e y) en parquet.

    Returns las rutas (x_path, y_path).
    """
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, _ = load_raw()
    X = preprocess(features)
    y = labels.sort_values(config.ID_COL).reset_index(drop=True)

    x_path = output_dir / "X.parquet"
    y_path = output_dir / "y.parquet"
    X.to_parquet(x_path, index=False)
    y.to_parquet(y_path, index=False)
    logger.info("Guardado X -> %s  |  y -> %s", x_path, y_path)
    return x_path, y_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset()
