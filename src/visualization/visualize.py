# -*- coding: utf-8 -*-
"""Visualizaciones del proyecto.

Gráficos reutilizables que se registran como artefactos de MLflow.
Se usa el backend no interactivo ``Agg`` para poder generar imágenes
desde scripts sin display.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_fold_rmse(folds: pd.DataFrame, model_name: str) -> plt.Figure:
    """Gráfico de barras del RMSE por fold de la validación cruzada temporal.

    Parameters
    ----------
    folds : DataFrame con columnas ``fold`` y ``rmse`` (salida de
        ``cross_validate_model``).
    model_name : nombre del algoritmo, usado en el título.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(folds["fold"].astype(str), folds["rmse"], color="#4c72b0")
    mean = folds["rmse"].mean()
    ax.axhline(mean, color="#c44e52", linestyle="--", linewidth=1.2,
               label=f"media = {mean:.5f}")
    ax.set_xlabel("Fold (TimeSeriesSplit)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE por fold — {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig
