# ml_routes.py
from flask import Blueprint, render_template, request, jsonify, current_app
import os
from datetime import datetime, timedelta

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

ml = Blueprint("ml", __name__, template_folder="templates")

MODEL_PATH = os.path.join("uploads", "models", "dino_shelf_model.joblib")


def _train_synthetic_model(n=4000, random_state=42):
    """Entrena un modelo sobre datos sintéticos basados en reglas de negocio."""
    rng = np.random.default_rng(random_state)
    beer_types = ["fresa", "mora"]
    packings = ["botella_vidrio", "lata", "barril"]

    X, y = [], []
    for _ in range(n):
        beer = rng.choice(beer_types)
        pack = rng.choice(packings)
        ferm = rng.integers(3, 21)      # días
        storage = rng.integers(7, 120)  # días
        temp = rng.uniform(2.0, 22.0)   # °C

        # Regla base + ajustes (misma lógica que tu simulación, con refinamiento leve)
        base = 90.0
        base += 5 if beer == "fresa" else 3
        base += 0.5 * float(ferm)
        base += (15.0 - float(temp)) * 1.2
        base += 5 if pack == "botella_vidrio" else (3 if pack == "lata" else 1)
        base += 0.15 * np.log1p(storage) * (storage ** 0.25)  # efecto suave del almacenamiento

        noise = rng.normal(0.0, 3.0)  # ruido pequeño
        target = float(np.clip(base + noise, 30.0, 270.0))   # vida útil en días (acotada)

        X.append([beer, pack, float(ferm), float(storage), float(temp)])
        y.append(target)

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
            ("num", StandardScaler(), [2, 3, 4]),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", ct),
            ("reg", RandomForestRegressor(n_estimators=250, random_state=random_state)),
        ]
    )
    model.fit(X, y)
    return model


def _ensure_model():
    """Carga el modelo si existe; si no, lo entrena y guarda."""
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    if model is None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model = _train_synthetic_model()
        joblib.dump(model, MODEL_PATH)
    return model


@ml.route("/ml", methods=["GET"])
def ml_page():
    # Renderiza la plantilla del formulario ML
    return render_template("ml.html", title="DinoAnalyticsML")


@ml.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}

    required = [
        "productionDate",
        "beerType",
        "fermentationDays",
        "storageDays",
        "temperature",
        "packaging",
    ]
    missing = [k for k in required if payload.get(k) in (None, "", [])]
    if missing:
        return jsonify({"ok": False, "error": f"Faltan campos: {', '.join(missing)}"}), 400

    # Parseo de fecha
    try:
        prod_dt = datetime.strptime(payload["productionDate"], "%Y-%m-%d").date()
    except Exception:
        return jsonify({"ok": False, "error": "Fecha inválida (usa AAAA-MM-DD)."}), 400

    # Features
    try:
        beer = str(payload["beerType"])
        pack = str(payload["packaging"])
        ferm = float(payload["fermentationDays"])
        stor = float(payload["storageDays"])
        temp = float(payload["temperature"])
    except Exception:
        return jsonify({"ok": False, "error": "Tipos de datos inválidos."}), 400

    # Cargar o crear el modelo (en caché del app para eficiencia)
    model = current_app.config.setdefault("_DINO_MODEL", _ensure_model())

    # Predicción de días de vida útil
    X = [[beer, pack, ferm, stor, temp]]
    try:
        days = float(model.predict(X)[0])
    except Exception:
        # Fallback determinístico (casi igual a tu lógica original)
        days = (
            90.0
            + (5.0 if beer == "fresa" else 3.0)
            + 0.5 * ferm
            + (15.0 - temp) * 1.2
            + (5.0 if pack == "botella_vidrio" else (3.0 if pack == "lata" else 1.0))
        )
        days = float(np.clip(days, 30.0, 270.0))

    # Regla de negocio: al menos fermentación + almacenamiento + 7 días de colchón
    min_days = ferm + stor + 7.0
    days = max(days, min_days)

    expiration = prod_dt + timedelta(days=int(round(days)))
    return jsonify(
        {
            "ok": True,
            "shelf_life_days": int(round(days)),
            "expiration_date_iso": expiration.isoformat(),
        }
    )
