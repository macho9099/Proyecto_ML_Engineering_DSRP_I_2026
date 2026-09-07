"""Script de ejecución: feature store con Feast (Reto ML 2).

Flujo completo del feature store para las features clave de `target_4`:

    1. build   : genera el parquet fuente (data/processed/feast_features.parquet)
    2. apply   : registra entidad + feature view en el registry local
    3. materialize : carga las features al online store (SQLite)
    4. train   : construye el dataset de entrenamiento con
                 `get_historical_features` (retrieval point-in-time) y evalúa
                 el modelo productivo (RandomForest) sobre esas features,
                 registrando el experimento en MLflow
    5. serve   : demuestra `get_online_features` para el último día de mercado

Uso:
    python scripts/run_feature_store.py            # flujo completo
    python scripts/run_feature_store.py --skip-train
    python scripts/run_feature_store.py --no-mlflow
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from feast import FeatureStore  # noqa: E402

from src import config  # noqa: E402
from src.features.build_features import build_feast_source  # noqa: E402
from src.models import tracking  # noqa: E402
from src.models.train_model import cross_validate_model, fit_full  # noqa: E402

REPO_PATH = ROOT / "feature_store"
FEATURE_SERVICE = "target4_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature store Feast (MITSUI)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Omite el entrenamiento con features históricas")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Desactiva el tracking del experimento en MLflow")
    return parser.parse_args()


def apply_and_materialize(store: FeatureStore) -> None:
    """Registra las definiciones y materializa al online store."""
    from feature_store.features import market_day, target4_features

    store.apply([market_day, target4_features])
    logging.info("Definiciones aplicadas: entidad 'market_day' + view '%s'",
                 FEATURE_SERVICE)
    store.materialize_incremental(end_date=datetime.utcnow())
    logging.info("Features materializadas al online store")


def train_from_store(store: FeatureStore, use_mlflow: bool) -> None:
    """Entrena el modelo productivo con features servidas por Feast."""
    from src.features.build_features import date_id_to_timestamp

    y = pd.read_parquet(config.PROCESSED_DATA_DIR / "y.parquet")
    entity_df = y[[config.ID_COL, config.TARGET]].copy()
    entity_df["event_timestamp"] = date_id_to_timestamp(entity_df[config.ID_COL])

    features = [f"{FEATURE_SERVICE}:{f.name}"
                for f in store.get_feature_view(FEATURE_SERVICE).features]
    training_df = store.get_historical_features(
        entity_df=entity_df, features=features
    ).to_df()
    training_df = training_df.sort_values(config.ID_COL).reset_index(drop=True)
    feature_cols = [f.split(":")[1] for f in features]
    logging.info("Dataset histórico desde Feast: %s (%d features)",
                 training_df.shape, len(feature_cols))

    X_f = training_df[[config.ID_COL, *feature_cols]]
    y_f = training_df[[config.TARGET]]
    agg, folds = cross_validate_model(X_f, y_f, model_name="random_forest")
    print("\n=== RandomForest con features servidas por Feast "
          f"({len(feature_cols)} features clave) ===")
    print(folds.to_string(index=False))
    print(f"\nrmse_mean={agg['rmse_mean']:.5f}±{agg['rmse_std']:.5f}  "
          f"r2_mean={agg['r2_mean']:.4f}  corr_mean={agg['corr_mean']:.4f}")

    if use_mlflow:
        pipeline = fit_full(X_f, y_f, model_name="random_forest")
        tracking.setup_mlflow()
        tracking.log_cv_run(
            "random_forest_feast", pipeline, agg, folds,
            extra_params={
                "target": config.TARGET,
                "feature_source": "feast",
                "feature_view": FEATURE_SERVICE,
                "n_features": len(feature_cols),
            },
        )


def serve_online_demo(store: FeatureStore) -> None:
    """Recupera del online store las features del último día de mercado."""
    source = pd.read_parquet(config.PROCESSED_DATA_DIR / "feast_features.parquet")
    last_day = int(source[config.ID_COL].max())
    sample = [f"{FEATURE_SERVICE}:{c}"
              for c in ("spread_ret", "spread_z20", "spread_vol20", "spread_mom20")
              if c in source.columns] or [f"{FEATURE_SERVICE}:spread_z20"]
    result = store.get_online_features(
        features=sample, entity_rows=[{config.ID_COL: last_day}]
    ).to_dict()
    print(f"\n=== Online serving (date_id={last_day}) ===")
    for key, values in result.items():
        print(f"  {key}: {values[0]}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    build_feast_source()
    store = FeatureStore(repo_path=str(REPO_PATH))
    apply_and_materialize(store)
    if not args.skip_train:
        train_from_store(store, use_mlflow=not args.no_mlflow)
    serve_online_demo(store)


if __name__ == "__main__":
    main()
