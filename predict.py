# --- predict.py ---
import joblib
import numpy as np
import pandas as pd

# --- Paso 1: Cargar el modelo y el escalador ---
print("Cargando modelo y escalador...")
model = joblib.load('beer_rating_predictor.pkl')
scaler = joblib.load('scaler.pkl')
print("Modelo y escalador cargados correctamente ✅")

# --- Paso 2: Definir las características en el mismo orden que el entrenamiento ---
features = [
    'ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body',
    'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
    'Fruits', 'Hoppy', 'Spices', 'Malty'
]

# --- Paso 3: Ingresar valores manualmente o desde un CSV ---
# Puedes cambiar estos valores de ejemplo:
new_data = pd.DataFrame([{
    'ABV': 5.5,
    'Min IBU': 15,
    'Max IBU': 35,
    'Astringency': 0.2,
    'Body': 0.7,
    'Alcohol': 0.6,
    'Bitter': 0.4,
    'Sweet': 0.5,
    'Sour': 0.1,
    'Salty': 0.0,
    'Fruits': 0.3,
    'Hoppy': 0.4,
    'Spices': 0.1,
    'Malty': 0.6
}])

print("\nDatos ingresados para predicción:")
print(new_data)

# --- Paso 4: Escalar los datos con el mismo escalador del entrenamiento ---
X_new_scaled = scaler.transform(new_data)

# --- Paso 5: Realizar la predicción ---
prediction = model.predict(X_new_scaled)

print("\n=== RESULTADO DE LA PREDICCIÓN ===")
print(f"Calificación general estimada (review_overall): {prediction[0]:.2f}")

# --- Paso 6: Guardar predicción en archivo CSV (opcional) ---
output = new_data.copy()
output['Predicted_review_overall'] = prediction
output.to_csv('prediction_result.csv', index=False)
print("\nPredicción guardada en 'prediction_result.csv' ✅")
