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
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Imprimir las primeras 5 filas del dataset
print(dataset.head())

# Imprime la primera fila
print(X.iloc[0])

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Calcular el error cuadrático medio (MSE)
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)

print("Error cuadratico medio:", mse)

# Calcular el coeficiente de determinación (R^2)
r2 = model.score(X_test, y_test)

print("Coeficiente de determinacion:", r2)

# Lógica del match making
def hacer_match(viaje_a_evaluar):
    # Hacer un dataframe a partir del array de entrada
    x_a_predecir = pd.DataFrame({   'ciudad_origen' : [viaje_a_evaluar[1]],
                                    'capacidad_maxima' : [viaje_a_evaluar[2]],
                                    'valoracion' : [viaje_a_evaluar[3]],
                                    'ciudad_destino' : [viaje_a_evaluar[4]],
                                    'peso_carga' : [viaje_a_evaluar[5]],
                                    'valor_carga' : [viaje_a_evaluar[6]],
                                    'dimension_carga' : [viaje_a_evaluar[7]],
                                    'requisitos' : [viaje_a_evaluar[8]]})

    # Añadir columna distancia entre ciudades
    x_a_predecir['distancia_entre_ciudades'] = x_a_predecir.apply(lambda row: calcular_distancia(row['ciudad_origen'], row['ciudad_destino']), axis=1)

    # Eliminar columna ciudad_origen y ciudad_destino
    x_a_predecir = x_a_predecir.drop(x_a_predecir.columns[[0, 3]], axis=1)
    
    print("Viaje a predecir:")
    print(x_a_predecir.iloc[0])

    # Predecir
    match = model.predict([x_a_predecir])
    return match

# Ejemplo de uso
viaje_a_evaluar = [1, "Santiago", 1000, 4, "Valparaiso", 100, 100, 100, 0]

print("Match:", hacer_match(viaje_a_evaluar))