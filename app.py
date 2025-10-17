import os
import io
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file
)
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

# --- Render de gráficos sin GUI ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------
# Configuración básica
# ----------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "DataSheet")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def create_app():
    app = Flask(
        __name__,
        template_folder="Templates"  # Tu carpeta se llama "Templates"
    )
    app.config["SECRET_KEY"] = "dev-key-change-this"
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

    # Estado simple en memoria (suficiente para dev)
    app.current_df = None
    app.current_name = None

    # Carga dataset por defecto si existe
    default_csv = os.path.join(DATA_FOLDER, "data.csv")
    if os.path.exists(default_csv):
        df, err = load_csv(default_csv)
        if df is not None:
            app.current_df = df
            app.current_name = "data.csv"
        else:
            print(f"[WARN] No se pudo cargar DataSheet/data.csv: {err}")

    # ------------------------
    # Utilidades
    # ------------------------
    def allowed_file(filename):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_df_or_404():
        if app.current_df is None:
            flash("Aún no hay un dataset cargado. Sube un archivo CSV.", "warning")
            return None
        return app.current_df

    # ------------------------
    # Rutas
    # ------------------------
    @app.route("/")
    def index():
        """Página de inicio: upload + resumen + preview."""
        df = app.current_df
        summary = None
        preview = None
        numeric_cols, categorical_cols = [], []

        if df is not None:
            summary = build_summary(df)
            preview = df.head(50).copy()
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        return render_template(
            "inicio.html",
            has_df=df is not None,
            filename=app.current_name,
            summary=summary,
            preview=preview,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )

    @app.route("/upload", methods=["POST"])
    def upload():
        """Carga de CSV y set como dataset activo."""
        if "file" not in request.files:
            flash("No se envió ningún archivo.", "danger")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("Selecciona un archivo CSV.", "danger")
            return redirect(url_for("index"))

        if file and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_name = f"{ts}_{safe_name}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], final_name)
            file.save(save_path)

            df, err = load_csv(save_path)
            if df is None:
                flash(f"No se pudo leer el CSV: {err}", "danger")
                return redirect(url_for("index"))

            app.current_df = df
            app.current_name = safe_name
            flash(f"Archivo '{safe_name}' cargado correctamente.", "success")
            return redirect(url_for("index"))

        flash("Formato no permitido. Sube un .csv", "danger")
        return redirect(url_for("index"))

    @app.route("/entendimiento")
    def entendimiento():
        """Página de EDA y gráficos."""
        df = get_df_or_404()
        if df is None:
            return redirect(url_for("index"))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        # Sugerimos la primera numérica si existe
        default_numeric = numeric_cols[0] if numeric_cols else None
        default_categorical = categorical_cols[0] if categorical_cols else None

        return render_template(
            "entendimiento.html",
            filename=app.current_name,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            default_numeric=default_numeric,
            default_categorical=default_categorical
        )

    @app.route("/api/column-data")
    def api_column_data():
        """
        Devuelve datos para Chart.js:
        - mode=hist  -> histograma columna numérica
        - mode=counts -> value_counts columna categórica
        """
        df = get_df_or_404()
        if df is None:
            return jsonify({"error": "No dataset"}), 400

        col = request.args.get("col")
        mode = request.args.get("mode", "hist")
        bins = int(request.args.get("bins", 15))

        if col not in df.columns:
            return jsonify({"error": "Columna no encontrada"}), 400

        series = df[col].dropna()

        if mode == "counts":
            vc = series.astype(str).value_counts().head(30)  # top 30
            labels = vc.index.tolist()
            values = vc.values.tolist()
            return jsonify({"labels": labels, "values": values})

        # histograma numérico
        if not np.issubdtype(series.dtype, np.number):
            return jsonify({"error": "La columna no es numérica"}), 400

        counts, edges = np.histogram(series, bins=bins)
        labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)]
        return jsonify({"labels": labels, "values": counts.tolist()})

    @app.route("/plot/corr.png")
    def plot_corr():
        """PNG con heatmap de correlación (numérico)."""
        df = get_df_or_404()
        if df is None:
            return "", 404

        num = df.select_dtypes(include="number")
        if num.empty:
            # Generar imagen vacía con texto
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "No hay columnas numéricas", ha="center", va="center")
            ax.axis("off")
        else:
            corr = num.corr(numeric_only=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr, cmap="viridis")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Matriz de correlación")

            # Mostrar valores encima
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}",
                            ha="center", va="center", color="white", fontsize=8)

            fig.tight_layout()

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @app.route("/download/preview.csv")
    def download_preview():
        """Descarga las primeras 1000 filas como CSV."""
        df = get_df_or_404()
        if df is None:
            return "", 404

        out = io.StringIO()
        df.head(1000).to_csv(out, index=False)
        mem = io.BytesIO(out.getvalue().encode("utf-8"))
        mem.seek(0)
        return send_file(
            mem,
            as_attachment=True,
            download_name="preview.csv",
            mimetype="text/csv"
        )

    return app


# ----------------------------------
# Helpers de datos
# ----------------------------------
def load_csv(path):
    """Lee CSV con detección flexible de separador y encoding."""
    try:
        # sep=None infiere delimitador (coma, punto y coma, tab, etc.)
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
        return df, None
    except Exception as e_utf8:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1", on_bad_lines="skip")
            return df, None
        except Exception as e2:
            return None, f"{e_utf8} / {e2}"

def build_summary(df: pd.DataFrame):
    """Resumen general del dataset."""
    summary = {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "faltantes_total": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "num_cols": df.select_dtypes(include="number").columns.tolist(),
        "cat_cols": df.select_dtypes(exclude="number").columns.tolist()
    }

    # Tipos de datos
    dtypes = df.dtypes.astype(str).to_dict()
    summary["dtypes"] = dtypes

    # Estadísticos rápidos numéricos
    if not df.select_dtypes(include="number").empty:
        stats = df.describe().T.reset_index().rename(columns={"index": "columna"})
        summary["stats_table"] = stats.to_dict(orient="records")
    else:
        summary["stats_table"] = []

    return summary


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    app = create_app()
    # host='0.0.0.0' si quieres probar desde otro equipo de la red
    app.run(debug=True)
