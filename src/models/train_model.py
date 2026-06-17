# -*- coding: utf-8 -*-
"""Entrenamiento y benchmark de modelos para el reto MITSUI Commodity Prediction.

Problema SIMPLIFICADO: regresión de un **único** target sobre series de
tiempo financieras:

    config.TARGET = "target_4"
        = "LME_AH_Close - JPX_Gold_Standard_Futures_Close" (lag 1)

Cada algoritmo se encapsula en un **scikit-learn Pipeline** (imputación +,
cuando aplica, escalado + estimador).

La evaluación usa **validación cruzada temporal** (`TimeSeriesSplit`): varios
folds donde cada uno entrena con el pasado y valida con el futuro inmediato.
Se reporta media ± desviación de cada métrica, más estable que un corte único.

Algoritmos del benchmark (ver MODEL_NAMES):
    - linear_regression       -> LinearRegression       (con StandardScaler)
    - decision_tree           -> DecisionTreeRegressor
    - hist_gradient_boosting  -> HistGradientBoostingRegressor

Funciones reutilizables:
    build_pipeline(model_name)        -> Pipeline
    train(X, y, model_name)           -> pipeline (split único) + métricas
    cross_validate_model(X, y, name)  -> métricas agregadas por CV temporal
    fit_full(X, y, model_name)        -> pipeline ajustado con todos los datos
    benchmark(X, y)                   -> compara todos los modelos por CV
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config

logger = logging.getLogger(__name__)

# Algoritmos disponibles en el benchmark
MODEL_NAMES = ["linear_regression", "decision_tree", "hist_gradient_boosting"]

# Nº de folds por defecto para la validación cruzada temporal
N_SPLITS = 5


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


def build_pipeline(
    model_name: str = "hist_gradient_boosting",
    select_k: int | None = None,
    rfe_step: float = 0.1,
) -> Pipeline:
    """Construye el Pipeline para `model_name`.

    Pasos: imputer [-> scaler] [-> rfe] -> modelo.

    Si `select_k` no es None, se inserta **RFE** (Recursive Feature Elimination)
    para quedarse con `select_k` features. RFE usa un ranker basado en
    importancias (DecisionTree) porque HistGradientBoosting no expone
    `feature_importances_`; `rfe_step` es la fracción de features eliminada en
    cada iteración (0.1 = 10%, más rápido que eliminar de a una).
    """
    estimator, needs_scaling = _make_estimator(model_name)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    if select_k is not None:
        ranker = DecisionTreeRegressor(max_depth=8, random_state=0)
        steps.append(("rfe", RFE(estimator=ranker, n_features_to_select=select_k, step=rfe_step)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _clean_xy(X: pd.DataFrame, y: pd.DataFrame, target: str):
    """Matriz de features y vector target alineados, sin filas con target NaN."""
    feats = _feature_cols(X)
    target_series = y[target]
    mask = target_series.notna().to_numpy()
    return X[feats].to_numpy()[mask], target_series.to_numpy()[mask]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas de capacidad predictiva (RMSE principal, + MAE, R^2, corr)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    valid_fraction: float = 0.2,
    target: str | None = None,
    select_k: int | None = None,
) -> tuple[Pipeline, dict]:
    """Entrena UN algoritmo con un corte temporal único (rápido, para uso ágil)."""
    target = target or config.TARGET
    Xv, yv = _clean_xy(X, y, target)
    cut = int(len(Xv) * (1 - valid_fraction))

    pipeline = build_pipeline(model_name, select_k=select_k)
    pipeline.fit(Xv[:cut], yv[:cut])
    m = _metrics(yv[cut:], pipeline.predict(Xv[cut:]))
    metrics = {"model": model_name, "target": target, "valid_rmse": m["rmse"], **m}
    logger.info("train(%s) split único: %s", model_name, metrics)
    return pipeline, metrics


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    n_splits: int = N_SPLITS,
    target: str | None = None,
    select_k: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Validación cruzada temporal de UN algoritmo.

    Returns
    -------
    agg : dict con media y desviación de cada métrica entre folds.
    folds : DataFrame con las métricas de cada fold.
    """
    target = target or config.TARGET
    Xv, yv = _clean_xy(X, y, target)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for k, (tr_idx, te_idx) in enumerate(tscv.split(Xv), start=1):
        pipe = build_pipeline(model_name, select_k=select_k)
        pipe.fit(Xv[tr_idx], yv[tr_idx])
        m = _metrics(yv[te_idx], pipe.predict(Xv[te_idx]))
        m.update({"fold": k, "n_train": len(tr_idx), "n_valid": len(te_idx)})
        rows.append(m)

    folds = pd.DataFrame(rows)
    agg = {"model": model_name, "target": target, "n_splits": n_splits}
    for metric in ("rmse", "mae", "r2", "corr"):
        agg[f"{metric}_mean"] = float(folds[metric].mean())
        agg[f"{metric}_std"] = float(folds[metric].std())
    logger.info("CV(%s): rmse=%.5f±%.5f r2=%.4f corr=%.4f",
                model_name, agg["rmse_mean"], agg["rmse_std"], agg["r2_mean"], agg["corr_mean"])
    return agg, folds


def fit_full(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str = "hist_gradient_boosting",
    target: str | None = None,
    select_k: int | None = None,
) -> Pipeline:
    """Ajusta el pipeline con TODOS los datos limpios (modelo final a guardar)."""
    target = target or config.TARGET
    Xv, yv = _clean_xy(X, y, target)
    pipeline = build_pipeline(model_name, select_k=select_k)
    pipeline.fit(Xv, yv)
    return pipeline


def benchmark(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_names: list[str] | None = None,
    n_splits: int = N_SPLITS,
    target: str | None = None,
    select_k: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Compara todos los algoritmos por validación cruzada temporal.

    Para cada modelo: evalúa por CV (media±std) y ajusta el pipeline final con
    todos los datos. Si `select_k` no es None, todos usan RFE para quedarse con
    `select_k` features.

    Returns
    -------
    results : DataFrame ordenado por rmse_mean (mejor primero).
    pipelines : dict {model_name: pipeline ajustado con todos los datos}.
    """
    target = target or config.TARGET
    model_names = model_names or MODEL_NAMES

    rows, pipelines = [], {}
    for name in model_names:
        agg, _ = cross_validate_model(
            X, y, model_name=name, n_splits=n_splits, target=target, select_k=select_k
        )
        rows.append(agg)
        pipelines[name] = fit_full(X, y, model_name=name, target=target, select_k=select_k)

    cols = ["model", "n_splits", "rmse_mean", "rmse_std", "mae_mean",
            "r2_mean", "r2_std", "corr_mean", "corr_std"]
    results = pd.DataFrame(rows)[cols].sort_values("rmse_mean").reset_index(drop=True)
    logger.info("Benchmark CV temporal (mejor por rmse_mean):\n%s", results.to_string(index=False))
    return results, pipelines


def save_model(pipeline: Pipeline, path: Path | None = None) -> Path:
    path = Path(path) if path else config.MODELS_DIR / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline guardado en %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.data.make_dataset import load_raw, prepare_features

    features, labels, _ = load_raw()
    X = prepare_features(features)
    results, pipelines = benchmark(X, labels)
    best = results.iloc[0]["model"]
    save_model(pipelines[best])
