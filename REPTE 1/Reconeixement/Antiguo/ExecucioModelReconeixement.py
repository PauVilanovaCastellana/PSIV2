import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model


# Cargar el modelo
model = load_model('modelo_caracteres.h5')

# Función para predecir el carácter a partir de una imagen
def predecir_caracter(imagen_ruta):
    # Cargar y procesar la imagen
    imagen = cv2.imread(imagen_ruta)
    imagen = cv2.resize(imagen, (150, 150))  # Redimensionar a 150x150
    imagen = imagen / 255.0  # Normalizar la imagen
    imagen = np.expand_dims(imagen, axis=0)  # Añadir dimensión para el batch

    # Hacer la predicción
    prediccion = model.predict(imagen)
    clase_predicha = np.argmax(prediccion, axis=1)

    return clase_predicha[0]

# Cargar etiquetas (clases) del modelo
etiquetas = sorted(os.listdir('caracters'))  # Obtener las clases en orden

# Ruta de la imagen que quieres predecir
ruta_imagen = 'caracters/D/D_0.png'  # Cambia esto a la ruta de tu imagen

# Realizar la predicción
indice_prediccion = predecir_caracter(ruta_imagen)
caracter_predicho = etiquetas[indice_prediccion]

print(f'El carácter predicho es: {caracter_predicho}')
