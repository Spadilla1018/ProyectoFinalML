# --- ml_routes.py ---
from flask import Blueprint, render_template, request
import joblib
import numpy as np
import os

ml_bp = Blueprint('ml_bp', __name__)

# --- Rutas del modelo y escalador ---
model_path = os.path.join('uploads', 'models', 'beer_rating_predictor.pkl')
scaler_path = os.path.join('uploads', 'models', 'scaler.pkl')

# --- Cargar modelo y escalador ---
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- Columnas que usa el modelo ---
features = [
    'ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body',
    'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
    'Fruits', 'Hoppy', 'Spices', 'Malty'
]

@ml_bp.route('/analytics', methods=['GET', 'POST'])
def analytics():
    prediction = None

    if request.method == 'POST':
        try:
            # Tomar los datos del formulario (en el mismo orden que features)
            input_data = [float(request.form[feature]) for feature in features]
            input_array = np.array([input_data])

            # Escalar los datos antes de predecir
            scaled_data = scaler.transform(input_array)

            # Realizar la predicción
            prediction = round(model.predict(scaled_data)[0], 2)

        except Exception as e:
            prediction = f"⚠️ Error en la predicción: {e}"

    # Renderizar plantilla
    return render_template('analytics.html', features=features, prediction=prediction)

