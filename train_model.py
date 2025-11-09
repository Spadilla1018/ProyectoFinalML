# --- train_model.py --- 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np  # 👈 agregado para calcular RMSE manualmente

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# ✅ Bloque corregido
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

# --- Paso 8: Importancia de características ---
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis")
plt.title("Importancia de Características (Random Forest)")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()
plt.show()

# --- Paso 9: Guardar el modelo y el escalador ---
joblib.dump(xgb_model, 'beer_rating_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelo y escalador guardados correctamente ✅")

# --- Paso 10: Visualizar predicciones ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='teal')
plt.xlabel("Valor Real (review_overall)")
plt.ylabel("Predicción del Modelo")
plt.title("Predicciones vs Valores Reales (XGBoost)")
plt.grid(True)
plt.show()
