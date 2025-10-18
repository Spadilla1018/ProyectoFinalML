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
from datetime import datetime, date
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SAEnum
import enum

import pymysql
import os
import io
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

BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "img")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = "cambia-esta-clave-por-una-bien-larga-y-secreta"

# Evitar cache de estáticos en dev (imágenes)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# --- MySQL (XAMPP)
DB_NAME = "ml_dashboard"
DB_USER = "root"
DB_PASS = ""
DB_HOST = "localhost"
DB_PORT = 3306

# Crear BD si no existe
conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, port=DB_PORT)
try:
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
finally:
    conn.close()

# SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Dirs
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

db = SQLAlchemy(app)

# ----------------------------------
# Login
# ----------------------------------
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

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ----------------------------------
# Modelos de producción
# ----------------------------------
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

# ----------------------------------
# Helpers
# ----------------------------------
def plato_from_sg(sg: float|None):
    if not sg: return None
    return (-616.868) + 1111.14*sg - 630.272*(sg**2) + 135.997*(sg**3)

def abv_estimate(og: float|None, fg: float|None):
    if not og or not fg: return None
    return max(0.0, (og - fg) * 131.25)

# ----------------------------------
# Bootstrap DB + seed
# ----------------------------------
with app.app_context():
    db.create_all()
    insp = inspect(db.engine)
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

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)
    return wrapper

# ----------------------------------
# Entendimiento (estado en memoria)
# ----------------------------------
CURRENT_DF = None
CURRENT_FILENAME = None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv"}

def split_columns(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    default_numeric = numeric_cols[0] if numeric_cols else None
    default_categorical = categorical_cols[0] if categorical_cols else None
    return numeric_cols, categorical_cols, default_numeric, default_categorical

# ----------------------------------
# Auth routes
# ----------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated: return redirect(url_for("index"))
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        if not name or not email or not password:
            flash("Completa todos los campos.", "warning"); return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Este correo ya está registrado.", "danger"); return redirect(url_for("register"))
        user = User(name=name, email=email); user.set_password(password)
        db.session.add(user); db.session.commit()
        flash("Cuenta creada. Ahora puedes iniciar sesión.", "success")
        return redirect(url_for("login"))
    return render_template("register.html", title="Crear cuenta")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated: return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Credenciales inválidas.", "danger"); return redirect(url_for("login"))
        login_user(user, remember=True)
        flash(f"¡Bienvenido, {user.name}!", "success")
        return redirect(request.args.get("next") or url_for("index"))
    return render_template("login.html", title="Iniciar sesión")

@app.post("/logout")
@login_required
def logout():
    logout_user(); flash("Sesión cerrada.", "success")
    return redirect(url_for("login"))

# ----------------------------------
# Admin usuarios
# ----------------------------------
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
        flash("Completa todos los campos.", "warning"); return redirect(url_for("admin_users"))
    if User.query.filter_by(email=email).first():
        flash("Ese correo ya existe.", "danger"); return redirect(url_for("admin_users"))
    u = User(name=name, email=email, is_admin=is_admin); u.set_password(password)
    db.session.add(u); db.session.commit(); flash("Usuario creado.", "success")
    return redirect(url_for("admin_users"))

# ----------------------------------
# Rutas principales + Entendimiento
# ----------------------------------
@app.get("/")
def index():
    return render_template("inicio.html", title="Inicio", filename=CURRENT_FILENAME)

@app.post("/upload")
@login_required
def upload():
    global CURRENT_DF, CURRENT_FILENAME
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("Selecciona un archivo .csv", "warning"); return redirect(url_for("index"))
    if not allowed_file(f.filename):
        flash("Formato no permitido (usa .csv).", "danger"); return redirect(url_for("index"))
    path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
    f.save(path)
    try:
        try:
            df = pd.read_csv(path, sep=";")
            if df.shape[1] == 1: df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path)
        CURRENT_DF = df; CURRENT_FILENAME = f.filename
        flash(f"Archivo {f.filename} cargado correctamente.", "success")
        return redirect(url_for("entendimiento"))
    except Exception as e:
        flash(f"Error leyendo CSV: {e}", "danger"); return redirect(url_for("index"))

@app.get("/entendimiento")
@login_required
def entendimiento():
    if CURRENT_DF is None:
        flash("No hay dataset cargado actualmente.", "warning"); return redirect(url_for("index"))
    numeric_cols, categorical_cols, dnum, dcat = split_columns(CURRENT_DF)
    return render_template("entendimiento.html",
        title="Entendimiento de Datos",
        filename=CURRENT_FILENAME,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        default_numeric=dnum,
        default_categorical=dcat)

@app.get("/api/column-data")
@login_required
def api_column_data():
    if CURRENT_DF is None: return jsonify({"error":"No hay dataset cargado."}), 400
    mode = request.args.get("mode",""); col = request.args.get("col","")
    if col not in CURRENT_DF.columns: return jsonify({"error": f"Columna no encontrada: {col}"}), 400
    if mode == "hist":
        try: bins = int(request.args.get("bins", 15))
        except ValueError: bins = 15
        s = pd.to_numeric(CURRENT_DF[col], errors="coerce").dropna()
        if s.empty: return jsonify({"error":"La columna no tiene datos numéricos válidos."}), 400
        counts, edges = np.histogram(s, bins=bins)
        labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(len(edges)-1)]
        return jsonify({"labels": labels, "values": counts.tolist()})
    if mode == "counts":
        s = CURRENT_DF[col].astype(str).fillna("NaN"); vc = s.value_counts().head(30)
        return jsonify({"labels": vc.index.tolist(), "values": vc.values.tolist()})
    return jsonify({"error":"Modo inválido. Usa 'hist' o 'counts'."}), 400

@app.get("/plot/corr")
@login_required
def plot_corr():
    if CURRENT_DF is None:
        flash("No hay dataset cargado actualmente.", "warning"); return redirect(url_for("index"))
    num_df = CURRENT_DF.select_dtypes(include=[np.number])
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(4,2)); ax.text(0.5,0.5,"Sin columnas numéricas", ha="center", va="center"); ax.axis("off")
    else:
        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(4,0.6*len(corr)), max(3,0.6*len(corr))))
        im = ax.imshow(corr, aspect="auto"); ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right"); ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_title("Matriz de correlación")
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format="png", dpi=160); plt.close(fig); buf.seek(0)
    return send_file(buf, mimetype="image/png")

# ----------------------------------
# Lotes / Lecturas / Des-alcoholización
# ----------------------------------
@app.get("/batches")
@login_required
def batches_list():
    batches = Batch.query.order_by(Batch.id.desc()).all()
    flavors = Flavor.query.order_by(Flavor.name).all()
    return render_template("batches.html", batches=batches, flavors=flavors, title="Lotes")

@app.post("/batches")
@login_required
def batches_create():
    code = request.form.get("code","").strip()
    flavor_id = int(request.form.get("flavor_id"))
    start_date_str = request.form.get("start_date")
    target_days = int(request.form.get("target_days", 14))
    volume_l = float(request.form.get("volume_l", 20))
    yeast = request.form.get("yeast","").strip()
    notes = request.form.get("notes","").strip()
    if not code or not start_date_str:
        flash("Código y fecha de inicio son obligatorios.", "warning"); return redirect(url_for("batches_list"))
    if Batch.query.filter_by(code=code).first():
        flash("Ese código de lote ya existe.", "danger"); return redirect(url_for("batches_list"))
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    b = Batch(code=code, flavor_id=flavor_id, start_date=start_dt, target_days=target_days, volume_l=volume_l, yeast=yeast, notes=notes)
    db.session.add(b); db.session.commit(); flash("Lote creado.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.get("/batches/<int:batch_id>")
@login_required
def batch_detail(batch_id):
    b = Batch.query.get_or_404(batch_id)
    sgs = [r.sg for r in sorted(b.readings, key=lambda r: r.ts) if r.sg]
    og = sgs[0] if sgs else None; fg = sgs[-1] if len(sgs) >= 1 else None
    abv = abv_estimate(og, fg)
    return render_template("batch_detail.html", b=b, og=og, fg=fg, abv=abv, title=f"Lote {b.code}")

@app.post("/batches/<int:batch_id>/reading")
@login_required
def reading_add(batch_id):
    b = Batch.query.get_or_404(batch_id)
    ts = datetime.strptime(request.form.get("ts"), "%Y-%m-%dT%H:%M")
    temp_c = request.form.get("temp_c"); sg = request.form.get("sg"); ph = request.form.get("ph")
    notes = request.form.get("notes","").strip()
    fr = FermentationReading(batch_id=b.id, ts=ts,
                             temp_c=float(temp_c) if temp_c else None,
                             sg=float(sg) if sg else None,
                             ph=float(ph) if ph else None,
                             notes=notes)
    db.session.add(fr); db.session.commit(); flash("Lectura agregada.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.post("/batches/<int:batch_id>/dealcoholize")
@login_required
def dealcoholize_add(batch_id):
    b = Batch.query.get_or_404(batch_id)
    day = int(request.form.get("day", b.day_of_fermentation()))
    method = request.form.get("method", "agent_addition")
    note = request.form.get("note","").strip(); efficacy = request.form.get("efficacy_estimate")
    step = DeAlcoholizationStep(batch_id=b.id, day=day, method=DeAlcoholizationMethod(method),
                                note=note, efficacy_estimate=float(efficacy) if efficacy else None)
    b.status = BatchStatus.dealcoholizing
    db.session.add(step); db.session.commit(); flash("Paso de des-alcoholización registrado.", "success")
    return redirect(url_for("batch_detail", batch_id=b.id))

@app.get("/api/batch/<int:batch_id>/series")
@login_required
def api_batch_series(batch_id):
    b = Batch.query.get_or_404(batch_id)
    rs = sorted(b.readings, key=lambda r: r.ts)
    labels = [r.ts.strftime("%d-%b %H:%M") for r in rs]
    temp = [r.temp_c for r in rs]; sg = [r.sg for r in rs]; ph = [r.ph for r in rs]
    og = next((x for x in sg if x), None)
    abv_curve = [abv_estimate(og, x) if og and x else None for x in sg]
    plato = [plato_from_sg(x) if x else None for x in sg]
    return jsonify({
        "labels": labels, "temp": temp, "sg": sg, "plato": plato, "ph": ph,
        "abv": abv_curve, "target_days": b.target_days, "start_date": b.start_date.strftime("%Y-%m-%d")
    })

# ----------------------------------
# Páginas públicas: Producto + PDFs
# ----------------------------------
@app.get("/producto")
def producto():
    return render_template("producto.html", title="Nuestro Producto")

@app.get("/producto/folleto")
def producto_folleto():
    """Folleto PDF: logos, objetivo, proceso, nutricional y QR a /producto."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 36; x = margin; y = height - margin

    # Logos
    udec_path = os.path.join(IMG_DIR, "udec_logo.png")
    loja_path = os.path.join(IMG_DIR, "loja_logo.png")
    logo_h = 50
    try:
        if os.path.exists(udec_path): c.drawImage(ImageReader(udec_path), x, y - logo_h, width=120, height=logo_h, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    try:
        if os.path.exists(loja_path): c.drawImage(ImageReader(loja_path), width - margin - 120, y - logo_h, width=120, height=logo_h, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    y -= (logo_h + 16)

    c.setFillColor(colors.HexColor("#065f46")); c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Cerveza Artesanal Sin Alcohol · Fresa & Mora"); y -= 24

    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Objetivo del producto"); y -= 14
    c.setFont("Helvetica", 11)
    for line in [
        "Ofrecer una bebida saludable e innovadora que mantenga el carácter de la cerveza artesanal,",
        "con fermentación controlada y un proceso de des-alcoholización que garantice un ABV < 0.5%."
    ]:
        c.drawString(x, y, line); y -= 14

    y -= 6; c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Proceso"); y -= 14
    c.setFont("Helvetica", 11)
    for line in [
        "1) Maceración y hervido con ingredientes seleccionados.",
        "2) Inoculación de levadura y fermentación controlada (14 días).",
        "3) Des-alcoholización con agente natural durante la fermentación.",
        "4) Acondicionado, control sensorial y estabilización.",
        "5) Envasado y almacenamiento en frío."
    ]:
        c.drawString(x, y, line); y -= 14

    y -= 6; c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Información nutricional (porción 330 ml)"); y -= 18
    table_x = x; table_y = y; col1_w = 220; col2_w = 120; row_h = 18
    rows = [("Energía","45 kcal"),("Carbohidratos","10 g"),("Azúcares","6 g"),("Proteínas","0.5 g"),("Grasas","0 g"),("Sodio","10 mg"),("Alcohol","< 0.5% ABV")]
    c.setFillColor(colors.HexColor("#e5f7ef")); c.rect(table_x, table_y - row_h, col1_w + col2_w, row_h, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10)
    c.drawString(table_x + 6, table_y - row_h + 5, "Componente"); c.drawString(table_x + col1_w + 6, table_y - row_h + 5, "Valor")
    y_cursor = table_y - row_h; c.setFont("Helvetica", 10)
    for label, value in rows:
        y_cursor -= row_h; c.setFillColor(colors.white)
        c.rect(table_x, y_cursor, col1_w + col2_w, row_h, fill=1, stroke=1)
        c.setFillColor(colors.black); c.drawString(table_x + 6, y_cursor + 5, label); c.drawString(table_x + col1_w + 6, y_cursor + 5, value)
    y = y_cursor - 24

    # QR a /producto
    try:
        producto_url = request.url_root.rstrip("/") + url_for("producto")
        qr_img = qrcode.make(producto_url); qr_buf = io.BytesIO(); qr_img.save(qr_buf, format="PNG"); qr_buf.seek(0)
        qr_size = 120
        c.drawImage(ImageReader(qr_buf), width - margin - qr_size, y - qr_size + 10, width=qr_size, height=qr_size, mask='auto')
        c.setFont("Helvetica", 9); c.drawRightString(width - margin, y - qr_size - 4, "Escanea para saber más")
    except Exception: pass

    c.setFillColor(colors.gray); c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, 24, "Proyecto académico — Universidad de Cundinamarca & Universidad de Loja — 0.0% ABV")
    c.showPage(); c.save()
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="Folleto_Cerveza_Sin_Alcohol.pdf")

@app.get("/producto/ficha")
def producto_ficha():
    """Ficha técnica PDF: especificaciones de proceso, rangos, QA, envasado + QR a esta ficha."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 36; x = margin; y = height - margin

    # Logos
    udec_path = os.path.join(IMG_DIR, "udec_logo.png")
    loja_path = os.path.join(IMG_DIR, "loja_logo.png")
    logo_h = 50
    try:
        if os.path.exists(udec_path): c.drawImage(ImageReader(udec_path), x, y - logo_h, width=120, height=logo_h, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    try:
        if os.path.exists(loja_path): c.drawImage(ImageReader(loja_path), width - margin - 120, y - logo_h, width=120, height=logo_h, preserveAspectRatio=True, mask='auto')
    except Exception: pass
    y -= (logo_h + 16)

    # Título
    c.setFillColor(colors.HexColor("#0f766e")); c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Ficha Técnica — Cerveza 0.0% Fresa & Mora"); y -= 22
    c.setFillColor(colors.black); c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Fecha de emisión: {datetime.now().strftime('%Y-%m-%d')}"); y -= 8
    c.drawString(x, y, "Versión: 1.0"); y -= 14

    # Especificaciones de proceso
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
    c.setFillColor(colors.HexColor("#eef2ff")); c.rect(table_x, table_y - row_h, col1_w + col2_w, row_h, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10)
    c.drawString(table_x + 6, table_y - row_h + 5, "Parámetro"); c.drawString(table_x + col1_w + 6, table_y - row_h + 5, "Valor")
    y_cursor = table_y - row_h; c.setFont("Helvetica", 10)
    for k, v in specs:
        y_cursor -= row_h; c.setFillColor(colors.white); c.rect(table_x, y_cursor, col1_w + col2_w, row_h, fill=1, stroke=1)
        c.setFillColor(colors.black); c.drawString(table_x + 6, y_cursor + 5, k); c.drawString(table_x + col1_w + 6, y_cursor + 5, v)
    y = y_cursor - 18

    # Ingredientes y alérgenos
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "2. Ingredientes y alérgenos"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "Agua tratada; Malta de cebada; Lúpulo; Levadura; Fresa natural; Mora natural.",
        "Alérgenos: contiene cereales que contienen gluten (cebada)."
    ]:
        c.drawString(x, y, line); y -= 14

    # Control de calidad (QA)
    y -= 2; c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "3. Controles de calidad (QA)"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "• Registro diario de temperatura, SG y pH durante la fermentación.",
        "• Verificación de ABV estimado al final (cálculo por OG/FG).",
        "• Evaluación sensorial: aroma a frutos rojos, acidez brillante, espuma estable.",
        "• Microbiología (si aplica): ausencia de contaminación (pruebas rápidas).",
    ]:
        c.drawString(x, y, line); y -= 14

    # Envasado / almacenamiento
    y -= 2; c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "4. Envasado y almacenamiento"); y -= 14
    c.setFont("Helvetica", 10)
    for line in [
        "Presentación: Botella 330 ml / Lata 355 ml.",
        "Pasteurización suave o estabilización en frío.",
        "Almacenamiento: 4–8 °C, evitar luz directa.",
        "Vida útil sugerida: 6 meses."
    ]:
        c.drawString(x, y, line); y -= 14

    # QR a esta ficha
    y -= 4
    try:
        ficha_url = request.url_root.rstrip("/") + url_for("producto_ficha")
        qr_img = qrcode.make(ficha_url); qr_buf = io.BytesIO(); qr_img.save(qr_buf, format="PNG"); qr_buf.seek(0)
        qr_size = 120
        c.drawImage(ImageReader(qr_buf), width - margin - qr_size, y - qr_size + 10, width=qr_size, height=qr_size, mask='auto')
        c.setFont("Helvetica", 9); c.drawRightString(width - margin, y - qr_size - 4, "Escanea esta ficha técnica")
    except Exception:
        pass

    # Pie
    c.setFillColor(colors.gray); c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, 24, "Proyecto académico — Universidad de Cundinamarca & Universidad de Loja — 0.0% ABV")
    c.showPage(); c.save()
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="Ficha_Tecnica_Cerveza_00_FresaMora.pdf")

# ----------------------------------
# Run
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
