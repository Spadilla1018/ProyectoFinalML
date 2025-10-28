# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def _mysql_uri():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "ml_dashboard")
    user = os.getenv("DB_USER", "root")
    pwd  = os.getenv("DB_PASSWORD", "")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}?charset=utf8mb4"

class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", _mysql_uri())
    SQLALCHEMY_TRACK_MODIFICATIONS = False
