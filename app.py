from flask import Flask
from flask import render_template, request
import pandas as pd
import os

# Asegúrate de tener estos módulos implementados en tu proyecto
# Si no existen, comenta o implementa funciones dummy para pruebas
try:
    import Relineal
except ImportError:
    Relineal = None
try:
    import Reg_Logis as ReLogistica
except ImportError:
    ReLogistica = None
try:
    import knn
except ImportError:
    knn = None

app = Flask(__name__)

# Variable global para almacenar las conclusiones actuales
conclusiones_actuales = None

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/index1')
def index1():
    # Página de caso de uso: Predicción de consumo energético
    return render_template('index1.html')

@app.route('/index2')
def index2():
    # Página de caso de uso: Energía solar
    return render_template('index2.html')

@app.route('/index3')
def index3():
    # Página de caso de uso: Eficiencia energética
    return render_template('index3.html')

@app.route('/index4')
def index4():
    # Página de caso de uso: Energía industrial
    return render_template('index4.html')

@app.route('/LR', methods=["GET", "POST"])
def LR():
    # Ejercicio de Regresión Lineal para consumo energético
    calculateResult = None
    if request.method == "POST" and Relineal:
        try:
            temperatura = float(request.form["temperatura"])
            ocupacion = float(request.form["ocupacion"])
            calculateResult = Relineal.predecir_consumo_energia(temperatura, ocupacion)
            import time
            time.sleep(0.1)
            Relineal.save_plot(temperatura, ocupacion, calculateResult)
        except ValueError:
            return "Por favor ingrese valores numéricos válidos"
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template("rl.html", result=calculateResult)

@app.route('/conceptos')
def conceptos():
    return render_template('conceptos.html')

@app.route('/conceptos_reg_logistica')
def conceptos_reg_logistica():
    return render_template('conceptos_reg_logistica.html')

# Cargar datos desde el archivo CSV (de energía sostenible)
data = None
try:
    data = pd.read_csv('./DataSheet/data.csv', delimiter=';')
except FileNotFoundError:
    print("Error: El archivo data.csv no se encontró.")
except Exception as e:
    print(f"Error al cargar datos: {e}")

@app.route('/ejercicio_reg_logistica', methods=['GET', 'POST'])
def ejercicio_reg_logistica():
    # Ejercicio de Regresión Logística para eficiencia energética
    result = None
    if request.method == 'POST' and ReLogistica:
        try:
            eficiencia = float(request.form['eficiencia'])
            renovable = float(request.form['renovable'])
            tipo_instalacion = request.form['tipo_instalacion'].lower()
            consumo = float(request.form['consumo'])

            entrada = pd.DataFrame([{
                "eficiencia_energetica": eficiencia,
                "porcentaje_renovable": renovable,
                "consumo_total": consumo,
                "tipo_instalacion": tipo_instalacion
            }])

            entrada = pd.get_dummies(entrada, columns=["tipo_instalacion"], drop_first=True)

            for col in ReLogistica.x.columns:
                if col not in entrada.columns:
                    entrada[col] = 0
            entrada = entrada[ReLogistica.x.columns]

            features = entrada.values[0]
            etiqueta, probabilidad = ReLogistica.predict_label(features)

            result = {
                "etiqueta": etiqueta,
                "probabilidad": probabilidad
            }

        except ValueError:
            result = {"error": "Por favor ingrese valores válidos"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

    return render_template('ejercicio_reg_logistica.html', result=result)

@app.route('/TiposAlgoritmos')
def tipos_algoritmos():
    return render_template('TiposAlgoritmos.html')

@app.route('/ejercicio_knn', methods=['GET', 'POST'])
def ejercicio_knn():
    global conclusiones_actuales
    metrics = None
    pred = None
    prob = None

    if request.method == 'POST' and knn:
        if 'train' in request.form:
            try:
                datos_nuevos = verificar_datos_nuevos()
                resultado_completo = knn.entrenar_modelo()
                metrics = {
                    "accuracy": resultado_completo["accuracy"],
                    "report": resultado_completo["report"],
                    "classes": resultado_completo["classes"]
                }
                conclusiones_actuales = resultado_completo["conclusiones"]
                with open("static/ultimo_entrenamiento.txt", "w") as f:
                    f.write(str(pd.Timestamp.now()))
                pred = request.form.get('pred')
                prob = request.form.get('prob')
            except Exception as e:
                metrics = {'error': f'Error entrenando: {e}'}
        elif 'predict' in request.form:
            try:
                edad = float(request.form['edad'])
                consumo = float(request.form['consumo'])
                renovable = float(request.form['renovable'])
                tipo_usuario_num = int(request.form['tipo_usuario'])
                threshold = float(request.form.get('threshold', 0.5))
                tipo_usuario_map = {1: "Residencial", 2: "Comercial", 3: "Industrial"}
                tipo_usuario = tipo_usuario_map.get(tipo_usuario_num, "Residencial")
                features = {
                    "Edad": edad,
                    "ConsumoEnergia": consumo,
                    "PorcentajeRenovable": renovable,
                    "TipoUsuario": tipo_usuario
                }
                pred, prob = knn.predict_label(features, threshold)
            except Exception as e:
                pred = f"Error: {e}"
                prob = None
    else:
        if conclusiones_actuales is None and knn and os.path.exists("static/knn_model.pkl"):
            try:
                conclusiones_actuales = knn.generar_conclusiones_desde_modelo_existente()
            except:
                conclusiones_actuales = None

    return render_template(
        'ejercicio_knn.html',
        metrics=metrics,
        pred=pred,
        prob=prob,
        conclusiones=conclusiones_actuales
    )

@app.route('/entendimiento')
def entendimiento():
    return render_template('entendimiento.html')

def verificar_datos_nuevos():
    archivo_datos = "DataSheet/Knn_data.csv"
    archivo_entrenamiento = "static/ultimo_entrenamiento.txt"
    if not os.path.exists(archivo_entrenamiento):
        return True
    if not os.path.exists(archivo_datos):
        return False
    try:
        mod_time_datos = os.path.getmtime(archivo_datos)
        with open(archivo_entrenamiento, "r") as f:
            ultimo_entrenamiento = pd.Timestamp(f.read().strip())
        return mod_time_datos > ultimo_entrenamiento.timestamp()
    except:
        return True

if __name__ == '__main__':
    app.run(debug=True, port=5000)