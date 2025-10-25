from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import inspect, text, func
from functools import wraps
from datetime import datetime, date
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SAEnum
from werkzeug.exceptions import HTTPException
from jinja2 import TemplateNotFound
import enum
import logging
import traceback

# --- stdlib / terceros
import os
import io
import csv
import json
import smtplib
from email.message import EmailMessage

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# ---- QR
import qrcode

# ---- ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import joblib

# -----------------------------------------------------------------------------
# Paths / App
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# En Render el disco persistente no está disponible; usa /tmp para archivos de trabajo
RUNTIME_DIR = os.path.join("/tmp", "dinobrew")
UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads")
IMG_DIR = os.path.join(STATIC_DIR, "img")  # tus imágenes estáticas siguen en /static
MODELS_DIR = os.path.join(UPLOAD_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.getenv("SECRET_KEY", "dev-change-me")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # evitar cache fuerte en dev
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Logging claro (Render muestra stdout)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = app.logger

# -----------------------------------------------------------------------------
# Base de Datos (Render-friendly)
#   - Si DATABASE_URL está definida (Postgres/MySQL/…): úsala
#   - Si no, usa SQLite en /tmp (efímero; suficiente para demo)
# -----------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    SQLITE_PATH = os.path.join(RUNTIME_DIR, "app.db")
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    log.info(f"Usando SQLite temporal en {SQLITE_PATH}")
else:
    log.info(f"Usando DATABASE_URL (producción): {DATABASE_URL}")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -----------------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, server_default='0')
    created_at = db.Column(db.DateTime, server_default=func.now())
    runs = relationship("MLRun", back_populates="user")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------------------------------------------------------------------
# Entidades de producción
# -----------------------------------------------------------------------------
class Flavor(db.Model):
    __tablename__ = "flavors"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)

class BatchStatus(enum.Enum):
    planned = "planned"
    fermenting = "fermenting"
    dealcoholizing = "dealcoholizing"
    finished = "finished"

class Batch(db.Model):
    __tablename__ = "batches"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(40), unique=True, nullable=False)
    flavor_id = db.Column(db.Integer, db.ForeignKey("flavors.id"), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    target_days = db.Column(db.Integer, nullable=False, default=14)
    volume_l = db.Column(db.Float, nullable=False, default=20.0)
    yeast = db.Column(db.String(120), nullable=True)
    status = db.Column(SAEnum(BatchStatus), nullable=False, default=BatchStatus.fermenting)
    notes = db.Column(db.Text, nullable=True)

    flavor = relationship("Flavor")
    readings = relationship("FermentationReading", back_populates="batch", cascade="all, delete-orphan")
    dealc_steps = relationship("DeAlcoholizationStep", back_populates="batch", cascade="all, delete-orphan")

    def day_of_fermentation(self):
        return (date.today() - self.start_date).days

class FermentationReading(db.Model):
    __tablename__ = "fermentation_readings"
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    ts = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    temp_c = db.Column(db.Float, nullable=True)
    sg = db.Column(db.Float, nullable=True)
    ph = db.Column(db.Float, nullable=True)
    notes = db.Column(db.String(200), nullable=True)
    batch = relationship("Batch", back_populates="readings")

class DeAlcoholizationMethod(enum.Enum):
    agent_addition = "agent_addition"
    vacuum_distillation = "vacuum_distillation"
    membrane = "membrane"

class DeAlcoholizationStep(db.Model):
    __tablename__ = "dealcoholization_steps"
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batches.id"), nullable=False)
    day = db.Column(db.Integer, nullable=False)
    method = db.Column(SAEnum(DeAlcoholizationMethod), nullable=False, default=DeAlcoholizationMethod.agent_addition)
    ts = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    note = db.Column(db.String(240), nullable=True)
    efficacy_estimate = db.Column(db.Float, nullable=True)
    batch = relationship("Batch", back_populates="dealc_steps")

# -----------------------------------------------------------------------------
# Registro de corridas ML
# -----------------------------------------------------------------------------
class MLRun(db.Model):
    __tablename__ = "ml_runs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    filename = db.Column(db.String(255), nullable=True)
    task = db.Column(db.String(32), nullable=False)  # 'classification' | 'regression'
    algorithm = db.Column(db.String(64), nullable=False)
    target = db.Column(db.String(255), nullable=False)

    features_json = db.Column(db.Text, nullable=False)
    metrics_json = db.Column(db.Text, nullable=False)
    schema_json = db.Column(db.Text, nullable=True)
    model_path = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())

    user = relationship("User", back_populates="runs")

# -----------------------------------------------------------------------------
# Mensajes de contacto
# -----------------------------------------------------------------------------
class ContactMessage(db.Model):
    __tablename__ = "contact_messages"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())
    ip = db.Column(db.String(64), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    emailed = db.Column(db.Boolean, nullable=False, server_default="0")
    user = relationship("User")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv"}

def split_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    default_numeric = numeric_cols[0] if numeric_cols else None
    default_categorical = categorical_cols[0] if categorical_cols else None
    return numeric_cols, categorical_cols, default_numeric, default_categorical

def infer_task_from_target(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.dropna().nunique()
        return "regression" if nunique > 10 else "classification"
    return "classification"

def build_pipeline(task: str, algorithm: str, X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_processor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_processor, num_cols),
            ("cat", categorical_processor, cat_cols)
        ]
    )

    if task == "regression":
        if algorithm == "linreg":
            model = LinearRegression()
        elif algorithm == "rf":
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        elif algorithm == "knn":
            model = KNeighborsRegressor(n_neighbors=5)
        else:
            raise ValueError("Algoritmo de regresión no soportado.")
    else:
        if algorithm == "logreg":
            model = LogisticRegression(max_iter=300)
        elif algorithm == "rf":
            model = RandomForestClassifier(n_estimators=300, random_state=42)
        elif algorithm == "knn":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError("Algoritmo de clasificación no soportado.")

    return Pipeline(steps=[("pre", pre), ("model", model)])

# -----------------------------------------------------------------------------
# Estado en memoria del dataset
# -----------------------------------------------------------------------------
CURRENT_DF = None
CURRENT_FILENAME = None

# -----------------------------------------------------------------------------
# Bootstrap DB + seed
# -----------------------------------------------------------------------------
with app.app_context():
    db.create_all()
    insp = inspect(db.engine)

    # endurecer tabla users
    cols = [c['name'] for c in insp.get_columns('users')]
    if 'is_admin' not in cols:
        db.session.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0;"))
        db.session.commit()
    if 'created_at' not in cols:
        db.session.execute(text("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP;"))
        db.session.commit()

    if not User.query.filter_by(email="admin@local").first():
        admin = User(name="Administrador", email="admin@local", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()

    if not Flavor.query.filter_by(name="Fresa & Mora").first():
        db.session.add(Flavor(name="Fresa & Mora"))
        db.session.commit()

    ml_cols = [c['name'] for c in insp.get_columns('ml_runs')]
    if 'schema_json' not in ml_cols:
        db.session.execute(text("ALTER TABLE ml_runs ADD COLUMN schema_json TEXT NULL;"))
        db.session.commit()

# -----------------------------------------------------------------------------
# Decoradores
# -----------------------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)
    return wrapper

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or not password:
            flash("Completa todos los campos.", "warning")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Este correo ya está registrado.", "danger")
            return redirect(url_for("register"))
        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Cuenta creada. Ahora puedes iniciar sesión.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", title="Crear cuenta")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Credenciales inválidas.", "danger")
            return redirect(url_for("login"))
        login_user(user, remember=True)
        flash(f"¡Bienvenido, {user.name}!", "success")
        return redirect(request.args.get("next") or url_for("index"))
    return render_template("login.html", title="Iniciar sesión")

@app.post("/logout")
@login_required
def logout():
    logout_user()
    flash("Sesión cerrada.", "success")
    return redirect(url_for("login"))

# -----------------------------------------------------------------------------
# Admin usuarios
# -----------------------------------------------------------------------------
@app.get("/admin/users")
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.id.desc()).all()
    return render_template("admin_users.html", users=users, title="Usuarios")

@app.post("/admin/users")
@login_required
@admin_required
def admin_users_create():
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    is_admin = bool(request.form.get("is_admin"))
    if not name or not email or not password:
        flash("Completa todos los campos.", "warning")
        return redirect(url_for("admin_users"))
    if User.query.filter_by(email=email).first():
        flash("Ese correo ya existe.", "danger")
        return redirect(url_for("admin_users"))
    u = User(name=name, email=email, is_admin=is_admin)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    flash("Usuario creado.", "success")
    return redirect(url_for("admin_users"))

# -----------------------------------------------------------------------------
# Rutas principales + EDA
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    try:
        return render_template("inicio.html", title="Inicio", filename=CURRent_FILENAME if False else CURRENT_FILENAME)
    except TemplateNotFound as e:
        log.error(f"Template no encontrado: {e.name}")
        return (
            f"Template <b>{e.name}</b> no encontrado. "
            f"Verifica que exista en <code>templates/</code> y se llame exactamente <code>inicio.html</code>.",
            500,
        )
    except Exception as e:
        log.error("Error en GET / -> %s\n%s", e, traceback.format_exc())
        return ("Ocurrió un error al cargar la página de inicio. Revisa los logs para la traza completa.", 500)

@app.post("/upload")
@login_required
def upload():
    global CURRENT_DF, CURRENT_FILENAME
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Selecciona un archivo .csv", "warning")
        return redirect(url_for("index"))
    if not allowed_file(f.filename):
        flash("Formato no permitido (usa .csv).", "danger")
        return redirect(url_for("index"))
    path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(path)
    try:
        try:
            df = pd.read_csv(path, sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path)
        CURRENT_DF = df
        CURRENT_FILENAME = f.filename
        session['dataset_path'] = path
        flash(f"Archivo {f.filename} cargado correctamente.", "success")
        return redirect(url_for("entendimiento"))
    except Exception as e:
        flash(f"Error leyendo CSV: {e}", "danger")
        return redirect(url_for("index"))

@app.get("/entendimiento")
@login_required
def entendimiento():
    if CURRENT_DF is None:
        flash("No hay dataset cargado actualmente.", "warning")
        return redirect(url_for("index"))
    numeric_cols, categorical_cols, dnum, dcat = split_columns(CURRENT_DF)
    all_cols = CURRENT_DF.columns.tolist()
    return render_template(
        "entendimiento.html",
        title="Entendimiento de Datos",
        filename=CURRENT_FILENAME,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        all_cols=all_cols,
        default_numeric=dnum,
        default_categorical=dcat
    )

@app.get("/api/column-data")
@login_required
def api_column_data():
    if CURRENT_DF is None:
        return jsonify({"error": "No hay dataset cargado."}), 400
    mode = request.args.get("mode", "")
    col = request.args.get("col", "")
    if col not in CURRENT_DF.columns:
        return jsonify({"error": f"Columna no encontrada: {col}"}), 400
    if mode == "hist":
        try:
            bins = int(request.args.get("bins", 15))
        except ValueError:
            bins = 15
        s = pd.to_numeric(CURRENT_DF[col], errors="coerce").dropna()
        if s.empty:
            return jsonify({"error": "La columna no tiene datos numéricos válidos."}), 400
        counts, edges = np.histogram(s, bins=bins)
        labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(len(edges) - 1)]
        return jsonify({"labels": labels, "values": counts.tolist()})
    if mode == "counts":
        s = CURRENT_DF[col].astype(str).fillna("NaN")
        vc = s.value_counts().head(30)
        return jsonify({"labels": vc.index.tolist(), "values": vc.values.tolist()})
    return jsonify({"error": "Modo inválido. Usa 'hist' o 'counts'."}), 400

@app.get("/plot/corr")
@login_required
def plot_corr():
    if CURRENT_DF is None:
        flash("No hay dataset cargado actualmente.", "warning")
        return redirect(url_for("index"))
    num_df = CURRENT_DF.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Sin columnas numéricas", ha="center", va="center")
        ax.axis("off")
    else:
        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(corr)), max(3, 0.6 * len(corr))))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.4 if len(corr) < 4 else 0.04)
        ax.set_title("Matriz de correlación")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

# -----------------------------------------------------------------------------
# Insights API
# -----------------------------------------------------------------------------
@app.route('/api/insights')
@login_required
def get_insights():
    try:
        if 'dataset_path' not in session:
            return jsonify({"error": "No hay dataset cargado"}), 400
        dataset_path = session['dataset_path']
        if not os.path.exists(dataset_path):
            return jsonify({"error": "El dataset no existe"}), 400
        df = pd.read_csv(dataset_path)
        insights = {}
        if 'edad' in df.columns:
            edad_promedio = df['edad'].mean()
            edad_min = df['edad'].min()
            edad_max = df['edad'].max()
            edad_rango = f"{int(edad_min)}-{int(edad_max)} años"
            if 'genero' in df.columns and 'gasto_mensual' in df.columns:
                top_spenders = df.nlargest(5, 'gasto_mensual')
                genero_comun = (top_spenders['genero'].mode().iloc[0] if not top_spenders.empty else "No disponible")
            else:
                genero_comun = "No disponible"
            if 'ingreso' in df.columns:
                ingreso_promedio = df['ingreso'].mean()
                if ingreso_promedio < 30000:
                    nivel_ingreso = "bajos"
                elif ingreso_promedio < 40000:
                    nivel_ingreso = "medios"
                else:
                    nivel_ingreso = "altos"
            else:
                nivel_ingreso = "no disponible"
            intereses = "productos frutales"
            if 'intereses' in df.columns:
                if not df['intereses'].empty:
                    intereses_comun = df['intereses'].mode().iloc[0]
                    if isinstance(intereses_comun, str):
                        intereses = intereses_comun
        else:
            edad_rango = "no disponible"
            genero_comun = "no disponible"
            nivel_ingreso = "no disponible"
            intereses = "no disponible"
        insights['customer_profile'] = {
            "age_range": edad_rango, "gender": genero_comun,
            "income_level": nivel_ingreso, "interests": intereses
        }
        if 'sabor_preferido' in df.columns:
            preferencias = df['sabor_preferido'].value_counts().reset_index()
            preferencias.columns = ['name', 'count']
            preferencias['percentage'] = (preferencias['count'] / preferencias['count'].sum() * 100).round(1)
            insights['preferences'] = preferencias[['name', 'percentage']].to_dict('records')
        else:
            insights['preferences'] = [{"name": "fresa", "percentage": 65.0}, {"name": "moras", "percentage": 35.0}]
        if 'sabor_preferido' in df.columns and 'precio_pagado' in df.columns:
            precios_optimos = []
            for sabor in df['sabor_preferido'].unique():
                sabor_df = df[df['sabor_preferido'] == sabor]
                precio_optimo = sabor_df['precio_pagado'].mean().round(0)
                precios_optimos.append({"product": sabor, "price": int(precio_optimo)})
            insights['optimal_prices'] = precios_optimos
        else:
            insights['optimal_prices'] = [{"product": "fresa", "price": 110}, {"product": "moras", "price": 125}]
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# Lotes / Lecturas / Des-alcoholización
# -----------------------------------------------------------------------------
def plato_from_sg(sg: float | None):
    if not sg:
        return None
    return (-616.868) + 1111.14*sg - 630.272*(sg**2) + 135.997*(sg**3)

def abv_estimate(og: float | None, fg: float | None):
    if not og or not fg:
        return None
    return max(0.0, (og - fg) * 131.25)

@app.get("/batches")
@login_required
def batches_list():
    batches = Batch.query.order_by(Batch.id.desc()).all()
    flavors = Flavor.query.order_by(Flavor.name).all()
    return render_template("batches.html", batches=batches, flavors=flavors, title="Lotes")

@app.post("/batches")
@login_required
def batches_create():
    code = request.form.get("code", "").strip()
    flavor_id = int(request.form.get("flavor_id"))
    start_date_str = request.form.get("start_date")
    target_days = int(request.form.get("target_days", 14))
    volume_l = float(request.form.get("volume_l", 20))
    yeast = request.form.get("yeast", "").strip()
    notes = request.form.get("notes", "").strip()
    if not code or not start_date_str:
        flash("Código y fecha de inicio son obligatorios.", "warning")
        return redirect(url_for("batches_list"))
    if Batch.query.filter_by(code=code).first():
        flash("Ese código de lote ya existe.", "danger")
        return redirect(url_for("batches_list"))
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    b = Batch(
        code=code, flavor_id=flavor_id, start_date=start_dt, target_days=target_days,
        volume_l=volume_l, yeast=yeast, notes=notes
    )
    db.session.add(b)
    db.session.commit()
    flash("Lote creado.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.get("/batches/<int:batch_id>")
@login_required
def batch_detail(batch_id):
    b = Batch.query.get_or_404(batch_id)
    sgs = [r.sg for r in sorted(b.readings, key=lambda r: r.ts) if r.sg]
    og = sgs[0] if sgs else None
    fg = sgs[-1] if len(sgs) >= 1 else None
    abv = abv_estimate(og, fg)
    return render_template("batch_detail.html", b=b, og=og, fg=fg, abv=abv, title=f"Lote {b.code}")

@app.post("/batches/<int:batch_id>/reading")
@login_required
def reading_add(batch_id):
    b = Batch.query.get_or_404(batch_id)
    ts = datetime.strptime(request.form.get("ts"), "%Y-%m-%dT%H:%M")
    temp_c = request.form.get("temp_c")
    sg = request.form.get("sg")
    ph = request.form.get("ph")
    notes = request.form.get("notes", "").strip()
    fr = FermentationReading(
        batch_id=b.id,
        ts=ts,
        temp_c=float(temp_c) if temp_c else None,
        sg=float(sg) if sg else None,
        ph=float(ph) if ph else None,
        notes=notes
    )
    db.session.add(fr)
    db.session.commit()
    flash("Lectura agregada.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.post("/batches/<int:batch_id>/dealcoholize")
@login_required
def dealcoholize_add(batch_id):
    b = Batch.query.get_or_404(batch_id)
    day = int(request.form.get("day", b.day_of_fermentation()))
    method = request.form.get("method", "agent_addition")
    note = request.form.get("note", "").strip()
    efficacy = request.form.get("efficacy_estimate")
    step = DeAlcoholizationStep(
        batch_id=b.id,
        day=day,
        method=DeAlcoholizationMethod(method),
        note=note,
        efficacy_estimate=float(efficacy) if efficacy else None
    )
    b.status = BatchStatus.dealcoholizing
    db.session.add(step)
    db.session.commit()
    flash("Paso de des-alcoholización registrado.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.get("/api/batch/<int:batch_id>/series")
@login_required
def api_batch_series(batch_id):
    b = Batch.query.get_or_404(batch_id)
    rs = sorted(b.readings, key=lambda r: r.ts)
    labels = [r.ts.strftime("%d-%b %H:%M") for r in rs]
    temp = [r.temp_c for r in rs]
    sg = [r.sg for r in rs]
    ph = [r.ph for r in rs]
    og = next((x for x in sg if x), None)
    abv_curve = [abv_estimate(og, x) if og and x else None for x in sg]
    plato = [plato_from_sg(x) if x else None for x in sg]
    return jsonify({
        "labels": labels, "temp": temp, "sg": sg, "plato": plato, "ph": ph,
        "abv": abv_curve, "target_days": b.target_days, "start_date": b.start_date.strftime("%Y-%m-%d")
    })

@app.post("/batches/<int:batch_id>/readings/upload")
@login_required
def readings_upload(batch_id):
    from csv import DictReader
    b = Batch.query.get_or_404(batch_id)
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Selecciona un CSV con columnas: ts,temp_c,sg,ph", "warning")
        return redirect(url_for("batch_detail", batch_id=b.id))
    try:
        tmp_path = os.path.join(UPLOAD_DIR, f"readings_{b.id}_{int(datetime.utcnow().timestamp())}.csv")
        f.save(tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as fh:
            dr = DictReader(fh)
            req_cols = {"ts", "temp_c", "sg", "ph"}
            if not req_cols.issubset(set(dr.fieldnames or [])):
                flash("El CSV debe tener columnas: ts,temp_c,sg,ph", "danger")
                return redirect(url_for("batch_detail", batch_id=b.id))
            added = 0
            for row in dr:
                ts = datetime.fromisoformat(row["ts"])
                temp = float(row["temp_c"]) if row["temp_c"] else None
                sg = float(row["sg"]) if row["sg"] else None
                ph = float(row["ph"]) if row["ph"] else None
                fr = FermentationReading(batch_id=b.id, ts=ts, temp_c=temp, sg=sg, ph=ph)
                db.session.add(fr)
                added += 1
            db.session.commit()
        os.remove(tmp_path)
        flash(f"{added} lecturas importadas.", "success")
    except Exception as e:
        flash(f"Error importando CSV: {e}", "danger")
    return redirect(url_for("batch_detail", batch_id=b.id))

# -----------------------------------------------------------------------------
# API de entrenamiento ML
# -----------------------------------------------------------------------------
@app.post("/api/ml/train")
@login_required
def api_ml_train():
    global CURRENT_DF, CURRENT_FILENAME
    if CURRENT_DF is None:
        return jsonify({"error": "No hay dataset cargado."}), 400
    try:
        payload = request.get_json(force=True)
        target = payload.get("target")
        features = payload.get("features", [])
        algorithm = payload.get("algorithm")  # 'linreg'|'logreg'|'rf'|'knn'
        task = payload.get("task")            # 'classification'|'regression'|None
        test_size = float(payload.get("test_size", 0.2))
        random_state = int(payload.get("random_state", 42))
        if not target or target not in CURRENT_DF.columns:
            return jsonify({"error": "Debes indicar una columna objetivo válida."}), 400
        if not features:
            features = [c for c in CURRENT_DF.columns if c != target]
        for fcol in features:
            if fcol not in CURRENT_DF.columns:
                return jsonify({"error": f"Columna de feature no existe: {fcol}"}), 400
        if algorithm not in {"linreg", "logreg", "rf", "knn"}:
            return jsonify({"error": "Algoritmo no soportado."}), 400
        df = CURRENT_DF.copy()
        y = df[target]
        X = df[features]
        if not task:
            task = infer_task_from_target(y)
        if task == "classification":
            if not pd.api.types.is_numeric_dtype(y):
                y = y.astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if task == "classification" else None
        )
        if task == "regression" and algorithm == "logreg":
            return jsonify({"error": "Regresión logística es para clasificación."}), 400
        if task == "classification" and algorithm == "linreg":
            return jsonify({"error": "Regresión lineal es para regresión."}), 400
        pipe = build_pipeline(task, algorithm, X_train)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = {}
        if task == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_test, y_pred)
            metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        else:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            metrics = {
                "accuracy": acc, "precision_macro": prec,
                "recall_macro": rec, "f1_macro": f1,
                "confusion_matrix": cm
            }
        schema = {}
        for c in features:
            schema[c] = "number" if pd.api.types.is_numeric_dtype(X[c]) else "string"
        run = MLRun(
            user_id=getattr(current_user, "id", None),
            filename=CURRENT_FILENAME or "",
            task=task, algorithm=algorithm, target=target,
            features_json=json.dumps(features), metrics_json=json.dumps(metrics),
            schema_json=json.dumps(schema), model_path=""
        )
        db.session.add(run)
        db.session.commit()
        model_path = os.path.join(MODELS_DIR, f"run_{run.id}.joblib")
        joblib.dump(pipe, model_path)
        run.model_path = model_path
        db.session.commit()
        return jsonify({
            "ok": True, "run_id": run.id, "task": task, "algorithm": algorithm,
            "target": target, "features": features, "metrics": metrics,
            "model_download": url_for("download_model", run_id=run.id)
        })
    except Exception as e:
        return jsonify({"error": f"Fallo entrenando el modelo: {str(e)}"}), 500

@app.get("/api/ml/runs")
@login_required
def api_ml_runs():
    q = MLRun.query
    if not getattr(current_user, "is_admin", False):
        q = q.filter((MLRun.user_id == current_user.id) | (MLRun.user_id.is_(None)))
    runs = q.order_by(MLRun.id.desc()).limit(50).all()
    data = []
    for r in runs:
        data.append({
            "id": r.id,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
            "filename": r.filename,
            "task": r.task,
            "algorithm": r.algorithm,
            "target": r.target,
            "features": json.loads(r.features_json),
            "metrics": json.loads(r.metrics_json),
            "download": url_for("download_model", run_id=r.id)
        })
    return jsonify(data)

@app.get("/download/model/<int:run_id>")
@login_required
def download_model(run_id):
    r = MLRun.query.get_or_404(run_id)
    if not os.path.exists(r.model_path):
        flash("Archivo de modelo no encontrado.", "danger")
        return redirect(url_for("entendimiento"))
    fname = f"modelo_{r.task}_{r.algorithm}_run{r.id}.joblib"
    return send_file(r.model_path, as_attachment=True, download_name=fname)

# -----------------------------------------------------------------------------
# Predicción en línea
# -----------------------------------------------------------------------------
@app.get("/predict/<int:run_id>")
@login_required
def predict_form(run_id):
    r = MLRun.query.get_or_404(run_id)
    if not os.path.exists(r.model_path):
        flash("Archivo de modelo no encontrado.", "danger")
        return redirect(url_for("entendimiento"))
    features = json.loads(r.features_json)
    schema = json.loads(r.schema_json or "{}")
    return render_template("predict.html", title=f"Predecir · Modelo {r.id}", run=r, features=features, schema=schema)

@app.post("/predict/<int:run_id>")
@login_required
def predict_submit(run_id):
    r = MLRun.query.get_or_404(run_id)
    if not os.path.exists(r.model_path):
        flash("Archivo de modelo no encontrado.", "danger")
        return redirect(url_for("entendimiento"))
    features = json.loads(r.features_json)
    schema = json.loads(r.schema_json or "{}")
    row = {}
    for f in features:
        val = request.form.get(f, "")
        if schema.get(f) == "number":
            try:
                row[f] = float(val) if val != "" else np.nan
            except ValueError:
                row[f] = np.nan
        else:
            row[f] = val
    X_infer = pd.DataFrame([row], columns=features)
    pipe = joblib.load(r.model_path)
    try:
        y_pred = pipe.predict(X_infer)
        y_proba = None
        try:
            y_proba = pipe.predict_proba(X_infer)[0].tolist()
        except Exception:
            pass
        return render_template(
            "predict.html",
            title=f"Predecir · Modelo {r.id}",
            run=r, features=features, schema=schema,
            input_values=row, prediction=y_pred[0], proba=y_proba
        )
    except Exception as e:
        flash(f"Error al predecir: {e}", "danger")
        return redirect(url_for("predict_form", run_id=r.id))

# -----------------------------------------------------------------------------
# Páginas públicas
# -----------------------------------------------------------------------------
@app.get("/producto")
def producto():
    return render_template("producto.html", title="Nuestro Producto")

@app.get("/producto/folleto")
def producto_folleto():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 36
    x = margin
    y = height - margin
    udec_path = os.path.join(IMG_DIR, "udec_logo.png")
    loja_path = os.path.join(IMG_DIR, "loja_logo.png")
    logo_h = 50
    try:
        if os.path.exists(udec_path):
            c.drawImage(ImageReader(udec_path), x, y - logo_h, width=120, height=logo_h,
                        preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    try:
        if os.path.exists(loja_path):
            c.drawImage(ImageReader(loja_path), width - margin - 120, y - logo_h, width=120, height=logo_h,
                        preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    y -= (logo_h + 16)
    c.setFillColor(colors.HexColor("#065f46"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Cerveza Artesanal Sin Alcohol · Fresa & Mora")
    y -= 24
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Objetivo del producto")
    y -= 14
    c.setFont("Helvetica", 11)
    for line in [
        "Ofrecer una bebida saludable e innovadora que mantenga el carácter de la cerveza artesanal,",
        "con fermentación controlada y un proceso de des-alcoholización que garantice un ABV < 0.5%."
    ]:
        c.drawString(x, y, line); y -= 14
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Proceso")
    y -= 14
    c.setFont("Helvetica", 11)
    for line in [
        "1) Maceración y hervido con ingredientes seleccionados.",
        "2) Inoculación de levadura y fermentación controlada (14 días).",
        "3) Des-alcoholización con agente natural durante la fermentación.",
        "4) Acondicionado, control sensorial y estabilización.",
        "5) Envasado y almacenamiento en frío."
    ]:
        c.drawString(x, y, line); y -= 14
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Información nutricional (porción 330 ml)")
    y -= 18
    table_x = x; table_y = y
    col1_w = 220; col2_w = 120; row_h = 18
    rows = [
        ("Energía", "45 kcal"), ("Carbohidratos", "10 g"), ("Azúcares", "6 g"),
        ("Proteínas", "0.5 g"), ("Grasas", "0 g"), ("Sodio", "10 mg"),
        ("Alcohol", "< 0.5% ABV"),
    ]
    c.setFillColor(colors.HexColor("#e5f7ef"))
    c.rect(table_x, table_y - row_h, col1_w + col2_w, row_h, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(table_x + 6, table_y - row_h + 5, "Componente")
    c.drawString(table_x + col1_w + 6, table_y - row_h + 5, "Valor")
    y_cursor = table_y - row_h
    c.setFont("Helvetica", 10)
    for label, value in rows:
        y_cursor -= row_h
        c.setFillColor(colors.white)
        c.rect(table_x, y_cursor, col1_w + col2_w, row_h, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.drawString(table_x + 6, y_cursor + 5, label)
        c.drawString(table_x + col1_w + 6, y_cursor + 5, value)
    y = y_cursor - 24
    try:
        producto_url = request.url_root.rstrip("/") + url_for("producto")
        qr_img = qrcode.make(producto_url)
        qr_buf = io.BytesIO()
        qr_img.save(qr_buf, format="PNG")
        qr_buf.seek(0)
        qr_size = 120
        c.drawImage(ImageReader(qr_buf), width - margin - qr_size, y - qr_size + 10,
                    width=qr_size, height=qr_size, mask='auto')
        c.setFont("Helvetica", 9)
        c.drawRightString(width - margin, y - qr_size - 4, "Escanea para saber más")
    except Exception:
        pass
    c.setFillColor(colors.gray)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, 24, "Proyecto académico — Universidad de Cundinamarca & Universidad de Loja — 0.0% ABV")
    c.showPage(); c.save(); buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="Folleto_Cerveza_Sin_Alcohol.pdf")

@app.get("/producto/ficha")
def producto_ficha():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 36; x = margin; y = height - margin
    udec_path = os.path.join(IMG_DIR, "udec_logo.png")
    loja_path = os.path.join(IMG_DIR, "loja_logo.png")
    logo_h = 50
    try:
        if os.path.exists(udec_path):
            c.drawImage(ImageReader(udec_path), x, y - logo_h, width=120, height=logo_h,
                        preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    try:
        if os.path.exists(loja_path):
            c.drawImage(ImageReader(loja_path), width - margin - 120, y - logo_h, width=120, height=logo_h,
                        preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    y -= (logo_h + 16)
    c.setFillColor(colors.HexColor("#0f766e"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Ficha Técnica — Cerveza 0.0% Fresa & Mora")
    y -= 22
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Fecha de emisión: {datetime.now().strftime('%Y-%m-%d')}")
    y -= 8; c.drawString(x, y, "Versión: 1.0"); y -= 14
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "1. Especificaciones de proceso"); y -= 16
    c.setFont("Helvetica", 10)
    specs = [
        ("Duración de fermentación", "14 días (objetivo)"),
        ("Temperatura fermentación", "18–22 °C"),
        ("Rango pH objetivo", "4.0 – 4.4"),
        ("OG (estimada)", "1.040 – 1.048"),
        ("FG (objetivo)", "1.008 – 1.014"),
        ("ABV (objetivo)", "< 0.5%"),
        ("Método de des-alcoholización", "Aditivo natural durante fermentación"),
        ("Saborizantes", "Fresa y Mora naturales"),
    ]
    col1_w, col2_w, row_h = 220, 200, 18
    table_x, table_y = x, y
    c.setFillColor(colors.HexColor("#eef2ff"))
    c.rect(table_x, table_y - row_h, col1_w + col2_w, row_h, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10)
    c.drawString(table_x + 6, table_y - row_h + 5, "Parámetro")
    c.drawString(table_x + col1_w + 6, table_y - row_h + 5, "Valor")
    y_cursor = table_y - row_h; c.setFont("Helvetica", 10)
    for k, v in specs:
        y_cursor -= row_h
        c.setFillColor(colors.white)
        c.rect(table_x, y_cursor, col1_w + col2_w, row_h, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.drawString(table_x + 6, y_cursor + 5, k)
        c.drawString(table_x + col1_w + 6, y_cursor + 5, v)
    y = y_cursor - 18
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "2. Ingredientes y alérgenos"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "Agua tratada; Malta de cebada; Lúpulo; Levadura; Fresa natural; Mora natural.",
        "Alérgenos: contiene cereales que contienen gluten (cebada)."
    ]:
        c.drawString(x, y, line); y -= 14
    y -= 2; c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "3. Controles de calidad (QA)"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "• Registro diario de temperatura, SG y pH durante la fermentación.",
        "• Verificación de ABV estimado al final (cálculo por OG/FG).",
        "• Evaluación sensorial: aroma a frutos rojos, acidez brillante, espuma estable.",
        "• Microbiología (si aplica): ausencia de contaminación (pruebas rápidas).",
    ]:
        c.drawString(x, y, line); y -= 14
    y -= 2; c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "4. Envasado y almacenamiento"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "Presentación: Botella 330 ml / Lata 355 ml.",
        "Pasteurización suave o estabilización en frío.",
        "Almacenamiento: 4–8 °C, evitar luz directa.",
        "Vida útil sugerida: 6 meses."
    ]:
        c.drawString(x, y, line); y -= 14
    y -= 4
    try:
        ficha_url = request.url_root.rstrip("/") + url_for("producto_ficha")
        qr_img = qrcode.make(ficha_url)
        qr_buf = io.BytesIO()
        qr_img.save(qr_buf, format="PNG"); qr_buf.seek(0)
        qr_size = 120
        c.drawImage(ImageReader(qr_buf), width - margin - qr_size, y - qr_size + 10,
                    width=qr_size, height=qr_size, mask='auto')
        c.setFont("Helvetica", 9)
        c.drawRightString(width - margin, y - qr_size - 4, "Escanea esta ficha técnica")
    except Exception:
        pass
    c.setFillColor(colors.gray); c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, 24, "Proyecto académico — Universidad de Cundinamarca & Universidad de Loja — 0.0% ABV")
    c.showPage(); c.save(); buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="Ficha_Tecnica_Cerveza_00_FresaMora.pdf")

@app.get("/nosotros")
def nosotros():
    return render_template("nosotros.html", title="Nosotros")

# -----------------------------------------------------------------------------
# Envío de email (opcional)
# -----------------------------------------------------------------------------
def send_contact_email(cm: ContactMessage) -> bool:
    enabled = os.getenv("MAIL_ENABLED", "1") == "1"
    if not enabled:
        return False
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    use_tls = os.getenv("SMTP_USE_TLS", "1") == "1"
    to_addr = os.getenv("MAIL_TO", "contacto@dinobrew.com")
    if not smtp_host or not to_addr:
        return False
    msg = EmailMessage()
    msg["Subject"] = f"[Dinobrew] {cm.subject}"
    msg["From"] = smtp_user or to_addr
    msg["To"] = to_addr
    body = (
        f"Nuevo mensaje de contacto\n\n"
        f"Nombre: {cm.name}\n"
        f"Email: {cm.email}\n"
        f"Asunto: {cm.subject}\n"
        f"IP: {cm.ip or '-'}\n"
        f"Usuario ID: {cm.user_id or '-'}\n"
        f"Fecha: {cm.created_at}\n\n"
        f"Mensaje:\n{cm.message}\n"
    )
    msg.set_content(body)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            if use_tls:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as e:
        app.logger.error(f"SMTP error: {e}")
        return False

# -----------------------------------------------------------------------------
# Contáctenos
# -----------------------------------------------------------------------------
@app.route("/contactenos", methods=["GET", "POST"])
def contactenos():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        if not all([name, email, subject, message]):
            flash("Completa todos los campos.", "warning")
            return redirect(url_for("contactenos", _anchor="form"))
        cm = ContactMessage(
            name=name, email=email, subject=subject, message=message,
            ip=request.headers.get("X-Forwarded-For", request.remote_addr),
            user_id=getattr(current_user, "id", None)
        )
        db.session.add(cm); db.session.commit()
        sent = send_contact_email(cm)
        if sent:
            cm.emailed = True; db.session.commit()
            flash("¡Gracias! Tu mensaje fue enviado y registrado correctamente.", "success")
        else:
            flash("Tu mensaje fue guardado. (Aviso: el envío de correo no se pudo completar).", "warning")
        return redirect(url_for("contactenos", _anchor="form"))
    return render_template("contactenos.html", title="Contáctenos")

# -----------------------------------------------------------------------------
# Admin de mensajes de contacto
# -----------------------------------------------------------------------------
@app.get("/admin/contactos")
@login_required
@admin_required
def admin_contactos():
    msgs = ContactMessage.query.order_by(ContactMessage.id.desc()).limit(500).all()
    return render_template("admin_contactos.html", title="Contactos", msgs=msgs)

@app.get("/admin/contactos/export.csv")
@login_required
@admin_required
def admin_contactos_export():
    msgs = ContactMessage.query.order_by(ContactMessage.id.asc()).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","created_at","name","email","subject","message","ip","user_id","emailed"])
    for m in msgs:
        w.writerow([
            m.id, m.created_at, m.name, m.email, m.subject,
            m.message.replace("\n"," ").strip(), m.ip, m.user_id, int(bool(m.emailed))
        ])
    mem = io.BytesIO(buf.getvalue().encode("utf-8")); mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="contact_messages.csv")

# -----------------------------------------------------------------------------
# Salud / Errores
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    try:
        db.session.execute(text("SELECT 1"))
        return {"ok": True, "db": "up"}, 200
    except Exception as e:
        return {"ok": False, "db": f"error: {e.__class__.__name__}"}, 500

@app.errorhandler(Exception)
def on_unhandled_exception(e):
    # Deja pasar HTTPException (404, etc.)
    if isinstance(e, HTTPException):
        return e
    log.error("Excepción no manejada:\n%s", traceback.format_exc())
    return ("Error interno. Revisa los logs de Render para la traza.", 500)

# -----------------------------------------------------------------------------
# Run (local)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
