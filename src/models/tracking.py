# -*- coding: utf-8 -*-
"""Administración de experimentos y modelos con MLflow.

Centraliza la interacción con MLflow para que el resto del código solo
llame funciones de alto nivel:

    setup_mlflow()                  -> configura tracking URI + experimento
    log_cv_run(...)                 -> registra un run de CV temporal completo
    register_model(run_id)          -> registra el modelo en el Model Registry
                                       y lo promueve con el alias 'production'

El destino del tracking se controla desde ``src.config`` / ``.env``:
con ``MLFLOW_TRACKING_URI`` vacío los runs quedan en ``<root>/mlruns``
(útil para desarrollo); apuntando a DagsHub quedan publicados en la nube.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)

AGG_METRICS = ("rmse", "mae", "r2", "corr")


def setup_mlflow() -> None:
    """Configura el tracking URI y el experimento activo.

    Sin ``MLFLOW_TRACKING_URI`` en el entorno, MLflow usa el backend local
    ``./mlruns`` (comportamiento por defecto de la librería).
    """
    if config.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    logger.info(
        "MLflow -> tracking: %s | experimento: %s",
        mlflow.get_tracking_uri(),
        config.MLFLOW_EXPERIMENT,
    )


def _pipeline_params(pipeline: Pipeline) -> dict:
    """Hiperparámetros primitivos del estimador final del pipeline."""
    model = pipeline.named_steps["model"]
    return {
        f"model__{key}": value
        for key, value in model.get_params().items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def log_cv_run(
    model_name: str,
    pipeline: Pipeline,
    agg: dict,
    folds: pd.DataFrame,
    extra_params: dict | None = None,
    input_example: np.ndarray | None = None,
) -> str:
    """Registra en MLflow un run completo de validación cruzada temporal.

    Se registran:
    - **Parámetros**: nombre del algoritmo, hiperparámetros del estimador y
      contexto del experimento (target, folds, selección de features, etc.).
    - **Métricas**: RMSE/MAE/R²/corr por fold (con ``step``) y agregadas.
    - **Artefactos**: métricas por fold en CSV y gráfico de RMSE por fold.
    - **Modelo**: el pipeline ajustado con todos los datos.

    Returns
    -------
    run_id : identificador del run.
    model_uri : URI ``models:/...`` del modelo loggeado (para registrarlo
        después en el Model Registry).
    """
    with mlflow.start_run(run_name=model_name) as run:
        params = {"model_name": model_name, **_pipeline_params(pipeline)}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        for row in folds.itertuples():
            for metric in AGG_METRICS:
                mlflow.log_metric(
                    f"fold_{metric}", float(getattr(row, metric)), step=int(row.fold)
                )
        mlflow.log_metrics(
            {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))}
        )

        with tempfile.TemporaryDirectory() as tmp:
            folds_path = Path(tmp) / "cv_folds.csv"
            folds.to_csv(folds_path, index=False)
            mlflow.log_artifact(str(folds_path))
            try:
                from src.visualization.visualize import plot_fold_rmse

                fig = plot_fold_rmse(folds, model_name)
                fig_path = Path(tmp) / "cv_rmse_por_fold.png"
                fig.savefig(fig_path, dpi=120, bbox_inches="tight")
                mlflow.log_artifact(str(fig_path))
            except Exception as exc:  # el gráfico no debe tumbar el run
                logger.warning("No se pudo generar el gráfico de folds: %s", exc)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        run_id = run.info.run_id
    logger.info("Run MLflow registrado: %s (%s)", model_name, run_id)
    return run_id, model_info.model_uri


def register_model(model_uri: str, alias: str = "production"):
    """Registra un modelo loggeado en el Model Registry y le asigna un alias.

    `model_uri` es la URI ``models:/...`` devuelta por `log_cv_run` (la forma
    ``runs:/<run_id>/model`` no resuelve en todos los servidores, p. ej.
    DagsHub). El alias ``production`` marca la versión productiva del modelo
    ``config.REGISTERED_MODEL_NAME``.
    """
    version = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    try:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            config.REGISTERED_MODEL_NAME, alias, version.version
        )
        logger.info(
            "Modelo '%s' v%s registrado con alias '%s'",
            config.REGISTERED_MODEL_NAME,
            version.version,
            alias,
        )
    except Exception as exc:  # algunos servidores no soportan aliases
        logger.warning("Modelo registrado, pero no se pudo asignar el alias: %s", exc)
    return version
