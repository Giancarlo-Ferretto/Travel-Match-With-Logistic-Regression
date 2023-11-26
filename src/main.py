import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from distances import calcular_distancia

# Cargar dataset
#  Columnas:
#   - id_camion
#   - ciudad_origen: [Santiago, Antofagasta, Valparaiso, Concepcion]
#   - capacidad_máxima_camión
#   - valoración_camión
#   - ciudad_destino: [Santiago, Antofagasta, Valparaiso, Concepcion]
#   - peso_carga
#   - valor_carga
#   - dimensión_carga
#   - requisitos especiales
#   - match
dataset = pd.read_csv("data/viajes.csv")

# Eliminar columna id_camion
dataset = dataset.drop(dataset.columns[[0]], axis=1)

# Añadir columna distancia entre ciudades
dataset["distancia_entre_ciudades"] = dataset.apply(lambda row: calcular_distancia(row[1], row[4]), axis=1)

# Eliminar columna ciudad_origen y ciudad_destino
dataset = dataset.drop(dataset.columns[[0, 3]], axis=1)

# Seleccionar las variables predictoras y la variable objetivo
X = dataset.drop(dataset.columns[[6]], axis=1)
y = dataset["match"]

# Imprimir las primeras 5 filas del dataset
# print(dataset.head())

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Lógica del match making
def hacer_match(viaje_a_evaluar):
    # Hacer un dataframe a partir del array de entrada
    columnas = ["ciudad_origen", "capacidad_maxima", "valoracion", "ciudad_destino",
            "peso_carga", "valor_carga", "dimension_carga", "requisitos"]

    x_a_predecir = pd.DataFrame([viaje_a_evaluar], columns=columnas)

    # Añadir columna distancia entre ciudades
    x_a_predecir["distancia_entre_ciudades"] = x_a_predecir.apply(lambda row: calcular_distancia(row["ciudad_origen"], row["ciudad_destino"]), axis=1)

    # Eliminar columna ciudad_origen y ciudad_destino
    x_a_predecir = x_a_predecir.drop(x_a_predecir.columns[[0, 3]], axis=1)
    
    print("Viaje a predecir:")
    print(x_a_predecir.iloc[0])

    # Predecir
    match = model.predict(x_a_predecir)

    # Calcular el error cuadrático medio (MSE)
    mse = np.mean((match - y_test) ** 2)

    print("[Match]: Error cuadratico medio:", mse)

    # Calcular el coeficiente de determinación (R^2)
    r2 = model.score(X_test, y_test)

    print("[Match]: Coeficiente de determinacion:", r2)

    # Retornar la predicción
    return match[0] == 1

# Ejemplo de uso
viaje_a_evaluar = ["Santiago", 1000, 4, "Valparaiso", 1000, 100000, 5, 0]

print("[Match]:", hacer_match(viaje_a_evaluar))