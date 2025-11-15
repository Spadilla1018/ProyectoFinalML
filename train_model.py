# --- train_model.py --- 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np  # para RMSE y ordenamientos

# --- RUTAS PARA GUARDAR GRÁFICAS EN static/img ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "img")
os.makedirs(IMG_DIR, exist_ok=True)

FEATURE_IMPORTANCE_PATH = os.path.join(IMG_DIR, "beer_feature_importance.png")
PRED_VS_REAL_PATH = os.path.join(IMG_DIR, "beer_pred_vs_real.png")

# --- Paso 1: Cargar el dataset ---
df = pd.read_csv('data/beer_profile_and_ratings.csv')

print("Primeras 5 filas:")
print(df.head())

print("\nInformación del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# --- Paso 2: Limpieza y selección de columnas ---
df = df.drop(['Beer_Name', 'Brewery'], axis=1, errors='ignore')

print("\nCantidad de valores nulos antes de limpiar:")
print(df.isnull().sum())
df = df.dropna()
print("\nCantidad de valores nulos después de limpiar:")
print(df.isnull().sum())

# --- Paso 3: Seleccionar columnas ---
features = [
    'ABV', 'Min IBU', 'Max IBU', 'Astringency', 'Body',
    'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
    'Fruits', 'Hoppy', 'Spices', 'Malty'
]
target = 'review_overall'

print("\nColumnas seleccionadas para el modelo:")
print(features)
print("\nColumna objetivo:", target)

# --- Paso 4: Dividir los datos ---
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTamaño de los conjuntos:")
print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)

# --- Paso 5: Escalado ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nDatos escalados correctamente ✅")

# --- Paso 6: Entrenamiento ---
print("\n=== ENTRENANDO MODELOS ===")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Modelo Random Forest entrenado correctamente ✅")

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
print("Modelo XGBoost entrenado correctamente ✅")

# --- Paso 7: Evaluación ---
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

print("\n--- Evaluación del Modelo ---")

# Random Forest
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest:")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R²: {r2_rf:.4f}")

# XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print("\nXGBoost:")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"R²: {r2_xgb:.4f}")

# --- Paso 8: Importancia de características (GUARDAR GRÁFICA) ---
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # de mayor a menor
sorted_features = [features[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_features)), sorted_importances)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.gca().invert_yaxis()  # para que la más importante quede arriba
plt.title("Importancia de características (Random Forest)")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()

plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=120, bbox_inches="tight")
print(f"\nGráfico de importancia de características guardado en: {FEATURE_IMPORTANCE_PATH}")
plt.show()
plt.close()

# --- Paso 9: Guardar el modelo y el escalador ---
joblib.dump(xgb_model, 'beer_rating_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelo y escalador guardados correctamente ✅")

# --- Paso 10: Visualizar y GUARDAR predicciones ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.6)
plt.xlabel("Valor real (review_overall)")
plt.ylabel("Predicción del modelo")
plt.title("Predicciones vs valores reales (XGBoost)")
plt.grid(True)

plt.savefig(PRED_VS_REAL_PATH, dpi=120, bbox_inches="tight")
print(f"Gráfico de predicciones vs valores reales guardado en: {PRED_VS_REAL_PATH}")
plt.show()
plt.close()
