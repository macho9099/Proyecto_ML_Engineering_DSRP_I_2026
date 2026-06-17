# -*- coding: utf-8 -*-
"""Entrenamiento y benchmark de modelos para el reto MITSUI Commodity Prediction.

Problema SIMPLIFICADO: regresión de un **único** target sobre series de
tiempo financieras:

    config.TARGET = "target_4"
        = "LME_AH_Close - JPX_Gold_Standard_Futures_Close" (lag 1)

Cada algoritmo se encapsula en un **scikit-learn Pipeline** que une el
preprocesamiento numérico (imputación y, cuando aplica, escalado) con el
estimador, de modo que las mismas transformaciones se apliquen en
entrenamiento e inferencia.

Algoritmos del benchmark (ver MODEL_NAMES):
    - linear_regression       -> LinearRegression       (con StandardScaler)
    - decision_tree           -> DecisionTreeRegressor
    - hist_gradient_boosting  -> HistGradientBoostingRegressor

Funciones reutilizables:
    build_pipeline(model_name) -> Pipeline
    train(X, y, model_name)    -> pipeline entrenado + métricas
    benchmark(X, y)            -> compara todos los modelos
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config

logger = logging.getLogger(__name__)

# Algoritmos disponibles en el benchmark
MODEL_NAMES = ["linear_regression", "decision_tree", "hist_gradient_boosting"]


def _feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != config.ID_COL]


def _make_estimator(model_name: str):
    """Devuelve (estimador, needs_scaling) para el algoritmo indicado."""
    if model_name == "linear_regression":
        return LinearRegression(), True
    if model_name == "decision_tree":
        return DecisionTreeRegressor(max_depth=8, random_state=0), False
    if model_name == "hist_gradient_boosting":
        return (
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
            False,
        )
    raise ValueError(f"Modelo desconocido: {model_name!r}. Opciones: {MODEL_NAMES}")


def build_pipeline(model_name: str = "hist_gradient_boosting") -> Pipeline:
    """Construye el Pipeline para `model_name`.

    Pasos:
      1. ``imputer`` -> SimpleImputer(median): red de seguridad para NaN.
      2. ``scaler``  -> StandardScaler: SOLO para modelos lineales.
      3. ``model``   -> el estimador del algoritmo elegido.
    """
    estimator, needs_scaling = _make_estimator(model_name)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _split_xy(X: pd.DataFrame, y: pd.DataFrame, target: str, valid_fraction: float):
    """Alinea X/y, descarta filas con target NaN y hace el split temporal."""
    feats = _feature_cols(X)
    target_series = y[target]
    valid_mask = target_series.notna().to_numpy()
    Xv = X[feats].to_numpy()[valid_mask]
    yv = target_series.to_numpy()[valid_mask]

    n = len(Xv)
    cut = int(n * (1 - valid_fraction))
    return Xv[:cut], Xv[cut:], yv[:cut], yv[cut:]


def _evaluate(pipeline: Pipeline, X_va, y_va, model_name: str, target: str) -> dict:
    """Métricas de capacidad predictiva sobre el conjunto de validación.

    - valid_rmse : raíz del error cuadrático medio (métrica principal).
    - valid_mae  : error absoluto medio (robusto a outliers).
    - valid_r2   : R^2, proporción de varianza explicada (<=1; 0 = no mejor
      que predecir la media; negativo = peor que la media).
    - valid_corr : correlación de Pearson pred vs. real.
    """
    preds = pipeline.predict(X_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    mae = float(mean_absolute_error(y_va, preds))
    r2 = float(r2_score(y_va, preds))
    corr = float(np.corrcoef(preds, y_va)[0, 1]) if len(y_va) > 1 else float("nan")
    return {
        "model": model_name,
        "target": target,
        "valid_rmse": rmse,
        "valid_mae": mae,
        "valid_r2": r2,
        "valid_corr": corr,
        "n_valid": int(len(y_va)),
    }


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    valid_fraction: float = 0.2,
    target: str | None = None,
) -> tuple[Pipeline, dict]:
    """Entrena UN algoritmo para `target` con split temporal (sin barajar)."""
    target = target or config.TARGET
    X_tr, X_va, y_tr, y_va = _split_xy(X, y, target, valid_fraction)

    logger.info(
        "Entrenando %s | target=%s (%s): train=%s valid=%s",
        model_name, target, config.TARGET_DEFINITION, X_tr.shape, X_va.shape,
    )
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_tr, y_tr)

    metrics = _evaluate(pipeline, X_va, y_va, model_name, target)
    logger.info("Validación: %s", metrics)
    return pipeline, metrics


def benchmark(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_names: list[str] | None = None,
    valid_fraction: float = 0.2,
    target: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Entrena y compara todos los algoritmos sobre el mismo split.

    Returns
    -------
    results : DataFrame ordenado por valid_rmse (mejor primero).
    pipelines : dict {model_name: pipeline entrenado}.
    """
    target = target or config.TARGET
    model_names = model_names or MODEL_NAMES

    rows, pipelines = [], {}
    for name in model_names:
        pipe, metrics = train(X, y, model_name=name, valid_fraction=valid_fraction, target=target)
        rows.append(metrics)
        pipelines[name] = pipe

    results = pd.DataFrame(rows).sort_values("valid_rmse").reset_index(drop=True)
    logger.info("Benchmark (mejor por valid_rmse):\n%s", results.to_string(index=False))
    return results, pipelines


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
    results, pipelines = benchmark(X, labels)
    best = results.iloc[0]["model"]
    save_model(pipelines[best])
