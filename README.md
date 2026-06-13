# Proyecto_ML_Engineering_DSRP_I_2026

Modelo de machine learning para mercados financieros, basado en el reto de Kaggle
**MITSUI & CO. Commodity Prediction Challenge**.

## Problema

Regresión **multi-objetivo** sobre series de tiempo financieras:

- **Features** (`train.csv`): 1918 días × 557 columnas de precios de commodities
  (LME, JPX, US Stocks, FX).
- **Targets** (`train_labels.csv`): 424 objetivos (`target_0 … target_423`). Cada uno
  es el retorno log de un instrumento o del *spread* entre dos instrumentos
  (ver `target_pairs.csv`), a un *lag* de 1, 2, 3 ó 4 días (106 targets por lag).
- **Test** (`test.csv`): 91 días, incluye la columna `is_scored`.
- `lagged_test_labels/` y `kaggle_evaluation/`: utilidades de evaluación de Kaggle.

## Estructura del repositorio

```
├── notebooks/
│   ├── 1.0-preprocesamiento-de-datos.ipynb   # carga + limpieza + data/processed
│   └── 2.0-machine-learning-y-llms.ipynb     # modelado ML + exploración LLMs
├── data/                 # raw / interim / processed (no versionado)
├── artifacts/            # salidas del proyecto
│   ├── models/           # modelos entrenados (.pkl)
│   ├── reports/          # reportes (.html)
│   └── predictions/      # predicciones (.csv)
├── src/                  # módulo de código reusable
│   ├── config.py         # rutas centrales (lee RAW_DATA_DIR de .env)
│   ├── data/make_dataset.py
│   └── models/{train_model.py, predict_model.py}
└── scripts/              # scripts de ejecución
    ├── run_preprocessing.py
    ├── run_training.py
    └── run_prediction.py
```

## Configuración de los datos

Los datos crudos viven **fuera del repo** (no se versionan). Su ubicación se define
en `.env`:

```
RAW_DATA_DIR=D:/DataScienceCompetitions/MITSUI&CO.Commodity_Prediction_Challenge/mitsui-commodity-prediction-challenge
```

Verifica que las rutas resuelven correctamente:

```bash
python -m src.config
```

## Uso

```bash
pip install -r requirements.txt

# 1. Preprocesamiento -> data/processed/X.parquet, y.parquet
python scripts/run_preprocessing.py

# 2. Entrenamiento -> artifacts/models/model.pkl
python scripts/run_training.py

# 3. Predicción -> artifacts/predictions/preds.csv
python scripts/run_prediction.py
```

Generado a partir de [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/).
