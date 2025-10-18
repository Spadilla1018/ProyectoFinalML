from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import inspect, text, func
from functools import wraps
import pymysql
import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN GENERAL
# ================================
BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.secret_key = "cambia-esta-clave-por-una-bien-larga-y-secreta"

# --- Config de MySQL (XAMPP) ---
DB_NAME = "ml_dashboard"
DB_USER = "root"     # Por defecto en XAMPP
DB_PASS = ""         # Vacío (sin contraseña) por defecto
DB_HOST = "localhost"
DB_PORT = 3306

# Crea la BD si no existe (usando PyMySQL)
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, port=DB_PORT)
try:
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
finally:
    conn.close()

# Config SQLAlchemy (ya apuntando a la BD)
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

db = SQLAlchemy(app)

# ================================
# LOGIN MANAGER
# ================================
login_manager = LoginManager(app)
login_manager.login_view = "login"  # si no autenticado, redirige a /login
login_manager.login_message_category = "warning"

# ================================
# MODELO USUARIO
# ================================
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, server_default='0')
    created_at = db.Column(db.DateTime, server_default=func.now())

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================================
# Asegurar columnas y admin por defecto
# ================================
with app.app_context():
    db.create_all()
    insp = inspect(db.engine)
    cols = [c['name'] for c in insp.get_columns('users')]

    # Asegurar is_admin
    if 'is_admin' not in cols:
        db.session.execute(text("ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT 0;"))
        db.session.commit()
    # Asegurar created_at
    if 'created_at' not in cols:
        db.session.execute(text("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP;"))
        db.session.commit()

    # Crear admin por defecto si no existe
    if not User.query.filter_by(email="admin@local").first():
        admin = User(name="Administrador", email="admin@local", is_admin=True)
        admin.set_password("admin123")  # ¡cámbiala luego!
        db.session.add(admin)
        db.session.commit()

# ================================
# Decorador admin
# ================================
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)
    return wrapper

# ================================
# ESTADO SIMPLE EN MEMORIA (dataset)
# ================================
CURRENT_DF = None
CURRENT_FILENAME = None

# ================================
# HELPERS
# ================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv"}

def split_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    default_numeric = numeric_cols[0] if numeric_cols else None
    default_categorical = categorical_cols[0] if categorical_cols else None
    return numeric_cols, categorical_cols, default_numeric, default_categorical

# ================================
# RUTAS DE AUTENTICACIÓN
# ================================
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
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Credenciales inválidas.", "danger")
            return redirect(url_for("login"))
        login_user(user, remember=True)
        flash(f"¡Bienvenido, {user.name}!", "success")
        return redirect(url_for("index"))
    # GET -> renderiza login.html (que extiende base.html)
    return render_template("login.html", title="Iniciar sesión")


@app.post("/logout")
@login_required
def logout():
    logout_user()
    flash("Sesión cerrada.", "success")
    return redirect(url_for("login"))

# ================================
# RUTAS ADMIN
# ================================
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
    name = request.form.get("name","").strip()
    email = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
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

# ================================
# RUTAS PRINCIPALES
# ================================
@app.get("/")
def index():
    return render_template("inicio.html", title="Inicio", filename=CURRENT_FILENAME)

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
        # lector flexible: intenta ; y luego ,
        try:
            df = pd.read_csv(path, sep=";")
            if df.shape[1] == 1:
                df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path)

        CURRENT_DF = df
        CURRENT_FILENAME = f.filename
        flash(f"Archivo {f.filename} cargado correctamente.", "success")
        return redirect(url_for("entendimiento"))
    except Exception as e:
        flash(f"Error leyendo CSV: {e}", "danger")
        return redirect(url_for("index"))

@app.get("/entendimiento")
@login_required
def entendimiento():
    if CURRENT_DF is None:
        flash("Primero sube un CSV para analizar.", "warning")
        return redirect(url_for("index"))

    numeric_cols, categorical_cols, dnum, dcat = split_columns(CURRENT_DF)
    return render_template(
        "entendimiento.html",
        title="Entendimiento de Datos",
        filename=CURRENT_FILENAME,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        default_numeric=dnum,
        default_categorical=dcat,
    )

# ================================
# ENDPOINTS PARA GRÁFICOS/API
# ================================
@app.get("/api/column-data")
@login_required
def api_column_data():
    """
    - Histograma: /api/column-data?mode=hist&col=COL&bins=15
    - Conteos  : /api/column-data?mode=counts&col=COL
    """
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
        labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(len(edges)-1)]
        return jsonify({"labels": labels, "values": counts.tolist()})

    if mode == "counts":
        s = CURRENT_DF[col].astype(str).fillna("NaN")
        vc = s.value_counts().head(30)
        return jsonify({"labels": vc.index.tolist(), "values": vc.values.tolist()})

    return jsonify({"error": "Modo inválido. Usa 'hist' o 'counts'."}), 400

@app.get("/plot/corr")
@login_required
def plot_corr():
    """PNG con matriz de correlación de columnas numéricas."""
    if CURRENT_DF is None:
        flash("Primero sube un CSV para analizar.", "warning")
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Matriz de correlación")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.get("/download/preview")
@login_required
def download_preview():
    """Descarga un CSV con los primeros 200 registros del dataset."""
    if CURRENT_DF is None:
        flash("No hay dataset para descargar.", "warning")
        return redirect(url_for("index"))

    preview = CURRENT_DF.head(200)
    csv_bytes = preview.to_csv(index=False).encode("utf-8")
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="preview.csv",
    )

# ================================
# EJECUCIÓN
# ================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
