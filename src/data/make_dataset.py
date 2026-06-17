# -*- coding: utf-8 -*-
"""Carga, preprocesamiento y feature engineering (MITSUI Commodity Prediction).

Pipeline de datos:
    load_raw()           -> (features, labels, target_pairs)
    preprocess()         -> niveles de precio ordenados y sin NaN (ffill/bfill)
    engineer_features()  -> features derivadas (retornos, rezagos, rolling, ...)
    prepare_features()   -> preprocess + engineer (un solo paso reutilizable)
    build_dataset()      -> guarda X/y procesados en data/processed/

Feature engineering (orientado a series de tiempo financieras, SIN fuga de
información: toda feature en t usa sólo datos hasta t):
  - Retorno simple (pct_change) de TODAS las columnas de precio -> estacionario.
  - Para los instrumentos clave (los dos del spread) y el spread mismo:
      * rezagos del retorno (lags)
      * media móvil del retorno
      * volatilidad (std móvil del retorno)
      * momentum (cambio acumulado en w periodos)
      * z-score del nivel (cuán lejos está de su media móvil)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Hiperparámetros de feature engineering
LAGS = (1, 2, 3, 5)
WINDOWS = (5, 10, 20)
WARMUP = max(WINDOWS) + max(LAGS)  # filas iniciales a descartar (NaN por rolling/lag)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSV crudos de la competencia."""
    logger.info("Cargando datos crudos desde %s", config.RAW_DATA_DIR)
    features = pd.read_csv(config.TRAIN_CSV)
    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    pairs = pd.read_csv(config.TARGET_PAIRS_CSV)
    logger.info("features=%s labels=%s pairs=%s", features.shape, labels.shape, pairs.shape)
    return features, labels, pairs


def load_test() -> pd.DataFrame:
    """Carga test.csv (incluye la columna `is_scored`)."""
    return pd.read_csv(config.TEST_CSV)


def preprocess(features: pd.DataFrame) -> pd.DataFrame:
    """Ordena por date_id e imputa faltantes de los niveles (ffill/bfill)."""
    df = features.sort_values(config.ID_COL).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != config.ID_COL]
    df[feature_cols] = df[feature_cols].ffill().bfill()
    return df


def _returns(series: pd.Series) -> pd.Series:
    """Retorno simple; inf (división por ~0) -> NaN."""
    return series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def engineer_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Construye features derivadas a partir de los niveles de precio.

    Devuelve un DataFrame con `date_id` + features. Las primeras filas
    contendrán NaN (warmup de rolling/lags); se descartan en build_dataset.
    """
    df = prices.sort_values(config.ID_COL).reset_index(drop=True)
    price_cols = [c for c in df.columns if c != config.ID_COL]

    out = pd.DataFrame({config.ID_COL: df[config.ID_COL].to_numpy()})

    # 1) Retorno de TODAS las columnas de precio (estacionariedad)
    rets = df[price_cols].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    rets.columns = [f"ret_{c}" for c in price_cols]

    # 2) Features enriquecidas para instrumentos clave y el spread
    left, right = config.SPREAD_LEFT, config.SPREAD_RIGHT
    spread = df[left] - df[right]
    key_series = {left: df[left], right: df[right], "spread": spread}

    extra: dict[str, pd.Series] = {}
    for name, s in key_series.items():
        # retorno: pct_change para precios (>0); diff para el spread (puede ser <0)
        ret = s.diff() if name == "spread" else _returns(s)
        extra[f"{name}_ret"] = ret
        for L in LAGS:
            extra[f"{name}_ret_lag{L}"] = ret.shift(L)
        for w in WINDOWS:
            extra[f"{name}_ret_mean{w}"] = ret.rolling(w).mean()
            extra[f"{name}_ret_vol{w}"] = ret.rolling(w).std()
            extra[f"{name}_mom{w}"] = s.diff(w) if name == "spread" else s.pct_change(w, fill_method=None)
            roll_mean = s.rolling(w).mean()
            roll_std = s.rolling(w).std()
            extra[f"{name}_z{w}"] = (s - roll_mean) / roll_std

    out = pd.concat([out, rets, pd.DataFrame(extra)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    logger.info("engineer_features -> %s features", out.shape[1] - 1)
    return out


def prepare_features(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """preprocess + engineer en un solo paso (usado en train e inferencia)."""
    return engineer_features(preprocess(raw_prices))


def build_dataset(output_dir: Path | None = None, drop_warmup: bool = True) -> tuple[Path, Path]:
    """Genera y persiste el dataset con features (X) y labels (y) en parquet."""
    output_dir = Path(output_dir) if output_dir else config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, _ = load_raw()
    X = prepare_features(features)

    # Alinear labels con X por date_id (garantiza el mismo orden)
    y = X[[config.ID_COL]].merge(labels, on=config.ID_COL, how="left")

    if drop_warmup:
        X = X.iloc[WARMUP:].reset_index(drop=True)
        y = y.iloc[WARMUP:].reset_index(drop=True)

    x_path = output_dir / "X.parquet"
    y_path = output_dir / "y.parquet"
    X.to_parquet(x_path, index=False)
    y.to_parquet(y_path, index=False)
    logger.info("Guardado X=%s -> %s | y=%s -> %s", X.shape, x_path, y.shape, y_path)
    return x_path, y_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset()
