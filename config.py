# config.py
from pathlib import Path

# Carpeta base del proyecto (donde está app.py)
BASE_DIR = Path(__file__).resolve().parent

# Carpeta uploads y subcarpeta models (la que ya tienes)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Usaremos EXACTAMENTE esta carpeta: uploads/models
MODELS_DIR = UPLOADS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ruta por defecto para el modelo de vencimiento
EXPIRY_MODEL_PATH = MODELS_DIR / "expiry_model.joblib"
