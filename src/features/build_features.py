# -*- coding: utf-8 -*-
"""Construcción de features para el feature store (Feast).

El feature engineering principal (lags, retornos, medias móviles) vive en
``src/data/make_dataset.py``. Este módulo materializa el subconjunto de
**features clave del target** (instrumentos del spread + spread) en el formato
que Feast necesita como *offline source*: una tabla parquet con la entidad
(``date_id``) y una columna ``event_timestamp``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# date_id es un entero secuencial (1 día = 1 paso); Feast exige un timestamp
# real, así que lo mapeamos a fechas sintéticas a partir de un día base.
BASE_DATE = pd.Timestamp("2020-01-01")

FEAST_SOURCE_PATH = config.PROCESSED_DATA_DIR / "feast_features.parquet"


def key_feature_columns(X: pd.DataFrame) -> list[str]:
    """Columnas de features derivadas de los instrumentos del spread."""
    prefixes = tuple(
        f"{k}_" for k in (config.SPREAD_LEFT, config.SPREAD_RIGHT, "spread")
    )
    return [c for c in X.columns if c.startswith(prefixes)]


def date_id_to_timestamp(date_id: pd.Series) -> pd.Series:
    """Mapea el índice temporal ``date_id`` a un timestamp sintético."""
    return BASE_DATE + pd.to_timedelta(date_id, unit="D")


def build_feast_source(out_path: Path | None = None) -> Path:
    """Genera el parquet fuente del feature store.

    Toma ``data/processed/X.parquet`` y guarda ``date_id``,
    ``event_timestamp`` y las features clave del target (51 columnas).
    """
    out_path = Path(out_path) if out_path else FEAST_SOURCE_PATH
    X = pd.read_parquet(config.PROCESSED_DATA_DIR / "X.parquet")
    cols = key_feature_columns(X)
    df = X[[config.ID_COL, *cols]].copy()
    df["event_timestamp"] = date_id_to_timestamp(df[config.ID_COL])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(
        "Fuente Feast: %s (%d filas x %d features)", out_path, len(df), len(cols)
    )
    return out_path
