# -*- coding: utf-8 -*-
"""Configuración central de rutas del proyecto.

Punto único de verdad para localizar los datos crudos del reto
MITSUI & CO. Commodity Prediction Challenge y los directorios de
salida (datos procesados y artefactos).

La ubicación de los datos crudos se resuelve, en orden:
    1. variable de entorno / .env  -> RAW_DATA_DIR
    2. valor por defecto (ubicación externa de la competencia)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Raíz del repositorio (este archivo vive en <root>/src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# --- Datos crudos de la competencia ------------------------------------------
# Por defecto apuntan a la carpeta externa de Kaggle; sobreescribir en .env
# (RAW_DATA_DIR=...) si los datos se mueven o se copian a data/raw.
_DEFAULT_RAW = (
    "D:/DataScienceCompetitions/MITSUI&CO.Commodity_Prediction_Challenge/"
    "mitsui-commodity-prediction-challenge"
)
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", _DEFAULT_RAW))

# --- Directorios del repositorio ---------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# --- Archivos de la competencia ----------------------------------------------
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TRAIN_LABELS_CSV = RAW_DATA_DIR / "train_labels.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
TARGET_PAIRS_CSV = RAW_DATA_DIR / "target_pairs.csv"
LAGGED_TEST_LABELS_DIR = RAW_DATA_DIR / "lagged_test_labels"

# --- Constantes del problema -------------------------------------------------
ID_COL = "date_id"

# Catálogo completo de targets de la competencia (referencia).
N_TARGETS = 424  # target_0 .. target_423
ALL_TARGET_COLS = [f"target_{i}" for i in range(N_TARGETS)]

# --- Problema SIMPLIFICADO: un solo objetivo ---------------------------------
# Modelamos un único target en lugar de los 424.
#   target_4 = "LME_AH_Close - JPX_Gold_Standard_Futures_Close" (lag 1)
TARGET = "target_4"
TARGET_DEFINITION = "LME_AH_Close - JPX_Gold_Standard_Futures_Close"
TARGET_LAG = 1


def ensure_output_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    for d in (
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        PREDICTIONS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"RAW_DATA_DIR   : {RAW_DATA_DIR}  (exists={RAW_DATA_DIR.exists()})")
    for name in ("TRAIN_CSV", "TRAIN_LABELS_CSV", "TEST_CSV", "TARGET_PAIRS_CSV"):
        p = globals()[name]
        print(f"{name:18}: {p}  (exists={p.exists()})")
