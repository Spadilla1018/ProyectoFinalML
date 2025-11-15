# predict.py
import os
import joblib
import numpy as np
import pandas as pd

# Rutas absolutas a los modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "beer_rating_predictor.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Orden de las características (DEBE coincidir con el entrenamiento)
FEATURES = [
    "ABV", "Min IBU", "Max IBU", "Astringency", "Body",
    "Alcohol", "Bitter", "Sweet", "Sour", "Salty",
    "Fruits", "Hoppy", "Spices", "Malty"
]

_model = None
_scaler = None


def _load_artifacts():
    """Carga modelo y escalador solo una vez (lazy loading)."""
    global _model, _scaler

    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)


def predict_beer_rating(values):
    """
    Realiza la predicción de review_overall para una cerveza.

    Parámetro:
      values puede ser:
        - lista/tupla de floats en el mismo orden que FEATURES, o
        - diccionario {nombre_feature: valor}

    Retorna:
      float con la calificación estimada.
    """
    _load_artifacts()

    if isinstance(values, dict):
        row = {feat: float(values.get(feat, 0.0)) for feat in FEATURES}
        df = pd.DataFrame([row])
    else:
        if len(values) != len(FEATURES):
            raise ValueError(
                f"Se esperaban {len(FEATURES)} valores, se recibieron {len(values)}."
            )
        df = pd.DataFrame([values], columns=FEATURES)

    X_scaled = _scaler.transform(df)
    pred = _model.predict(X_scaled)
    return float(pred[0])


if __name__ == "__main__":
    # Prueba rápida por consola
    ejemplo = {
        "ABV": 5.5,
        "Min IBU": 15,
        "Max IBU": 35,
        "Astringency": 0.2,
        "Body": 0.7,
        "Alcohol": 0.6,
        "Bitter": 0.4,
        "Sweet": 0.5,
        "Sour": 0.1,
        "Salty": 0.0,
        "Fruits": 0.3,
        "Hoppy": 0.4,
        "Spices": 0.1,
        "Malty": 0.6,
    }

    y_hat = predict_beer_rating(ejemplo)
    print(f"Predicción de review_overall (ejemplo): {y_hat:.2f}")
