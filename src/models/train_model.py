# -*- coding: utf-8 -*-
"""Entrenamiento de modelos para el reto MITSUI Commodity Prediction.

Problema: regresión multi-objetivo (424 targets) sobre series de tiempo
financieras. Se entrena un modelo por target (envuelto en MultiOutput) con
una división temporal (sin barajar) para respetar el orden cronológico.

El modelo se encapsula en un **scikit-learn Pipeline** que une el
preprocesamiento numérico (imputación) con el estimador, de modo que las
mismas transformaciones se apliquen automáticamente en entrenamiento y en
inferencia (un único objeto que se guarda/carga con joblib).

Funciones reutilizables:
    build_pipeline()  -> Pipeline (imputer -> MultiOutput(HGB))
    train(X, y)       -> pipeline entrenado + métricas de validación
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)


def _feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != config.ID_COL]


def build_pipeline(
    max_iter: int = 200,
    learning_rate: float = 0.05,
) -> Pipeline:
    """Construye el Pipeline de modelado.

    Pasos:
      1. ``imputer``  -> SimpleImputer(median): red de seguridad para NaN
         residuales (las features ya se rellenan en make_dataset.preprocess).
      2. ``model``    -> MultiOutputRegressor(HistGradientBoostingRegressor):
         un regresor por target. HGB es robusto, no requiere escalado y
         maneja relaciones no lineales.

    Para modelos lineales (p. ej. Ridge) basta con insertar un
    ``StandardScaler`` entre el imputer y el estimador.
    """
    base = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(base, n_jobs=-1)),
        ]
    )


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    valid_fraction: float = 0.2,
    **pipeline_kwargs,
) -> tuple[Pipeline, dict]:
    """Entrena el Pipeline con un split temporal (sin barajar).

    Los NaN en los targets (instrumentos sin cotización) se imputan a 0
    sólo para el ajuste; en la práctica conviene una máscara por target.
    """
    feats = _feature_cols(X)
    Xv = X[feats].to_numpy()
    yv = y[config.TARGET_COLS].to_numpy()

    n = len(Xv)
    cut = int(n * (1 - valid_fraction))
    X_tr, X_va = Xv[:cut], Xv[cut:]
    y_tr, y_va = yv[:cut], yv[cut:]

    # Imputación simple de NaN en targets para poder ajustar
    y_tr = np.nan_to_num(y_tr, nan=0.0)

    logger.info("Entrenando: train=%s valid=%s targets=%s", X_tr.shape, X_va.shape, yv.shape[1])
    pipeline = build_pipeline(**pipeline_kwargs)
    pipeline.fit(X_tr, y_tr)

    preds = pipeline.predict(X_va)
    metrics = {"valid_rmse": float(_masked_rmse(y_va, preds))}
    logger.info("Validación: %s", metrics)
    return pipeline, metrics


def _masked_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return float("nan")
    diff = (y_pred[mask] - y_true[mask]) ** 2
    return float(np.sqrt(diff.mean()))


def save_model(pipeline: Pipeline, path: Path | None = None) -> Path:
    path = Path(path) if path else config.MODELS_DIR / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline guardado en %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.data.make_dataset import load_raw, preprocess

    features, labels, _ = load_raw()
    X = preprocess(features)
    pipeline, _ = train(X, labels)
    save_model(pipeline)
