# -*- coding: utf-8 -*-
"""Definiciones del feature store (Feast) — Reto ML 2.

Entidad, fuente offline y feature view de las **features clave del target**
(`target_4`): las derivadas de los dos instrumentos del spread y del spread
mismo (51 columnas generadas por ``src/features/build_features.py``).

El esquema del feature view se construye dinámicamente leyendo el parquet
fuente, para no mantener 51 nombres a mano.
"""
from datetime import timedelta
from pathlib import Path

import pyarrow.parquet as pq
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64

SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "feast_features.parquet"
)

market_day = Entity(
    name="market_day",
    join_keys=["date_id"],
    value_type=ValueType.INT64,
    description="Día de mercado (índice temporal de la competencia MITSUI)",
)

target4_source = FileSource(
    name="target4_features_source",
    path=str(SOURCE_PATH),
    timestamp_field="event_timestamp",
)

_feature_names = [
    c
    for c in pq.read_schema(SOURCE_PATH).names
    if c not in ("date_id", "event_timestamp")
]

target4_features = FeatureView(
    name="target4_features",
    entities=[market_day],
    ttl=timedelta(days=3650),
    schema=[Field(name=c, dtype=Float64) for c in _feature_names],
    online=True,
    source=target4_source,
    description="Features clave del spread LME_AH vs JPX_Gold para target_4",
)
