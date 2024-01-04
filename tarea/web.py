import numpy as np
import pickle
from flask import Flask, request, render_template


def prediccion(input, sc, model):
    input_std = sc.transform(input)

    probabilidad    = model.predict_proba(input_std)

    indice = np.argmax(probabilidad)

    return (list_names[indice], probabilidad[0][indice])

def nombre_modelo(indice):
    switcher = {
        0: "Regresión logística",
        1: "Máquinas de soporte vectorial (SVM)",
        2: "Árboles de decisión",
        3: "K Vecinos más próximos (KNN)"
    }

    return switcher.get(indice)


# Cargamos los datos base

with open('tarea/datos/base.pck', 'rb') as f:
    list_names, sc = pickle.load(f)


# Carga de los modelos

model = []

# Regresión logistica

with open('tarea/datos/regresion.pck', 'rb') as f:
    model.append(pickle.load(f))

# SVM

with open('tarea/datos/svm.pck', 'rb') as f:
    model.append(pickle.load(f))

# Árbol de decisión

with open('tarea/datos/arbol.pck', 'rb') as f:
    model.append(pickle.load(f))

# K-NN

with open('tarea/datos/knn.pck', 'rb') as f:
    model.append(pickle.load(f))



app = Flask(__name__)

# Receptor de los datos del formulario para la predicción

@app.route('/predict', methods=['POST'])
def predict():
    algoritmo   = request.form.get('algoritmo')
    amplitud    = request.form.get('amplitud')
    longitud    = request.form.get('longitud')

    input = [float(amplitud), float(longitud)]


    tipo, probabilidad = prediccion([input], sc, model[int(algoritmo)])

    resultado = "Algoritmo: {}. Amplitud: {}. Longitud: {}. Tipo: Iris {}. Probabilidad: {:.2f}%".format(nombre_modelo(int(algoritmo)), amplitud, longitud, tipo, probabilidad * 100)
    
    return render_template('index.html', prediccion=resultado)    

# Página principal de la web

@app.route('/')
def home():
      return render_template('index.html')


# Usamos el puerto 8080 para nuestra web

if __name__ == '__main__':
    app.run(port=8080) 

# Orden para lanzar el web.py desde el shell de poetry
# poetry run python tarea/web.py
