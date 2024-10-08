# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:51:55 2024

@author: eduar
"""

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir la ruta de la carpeta de entrenamiento
train_dir_letras = 'C:/Users/eduar/OneDrive/Escritorio/Matriculas/elements'  # Cambia esto por tu ruta

# Generador de imágenes con preprocesamiento (normalización)
datagen_letras = ImageDataGenerator(rescale=1.0/255.0)

# Cargar datos de entrenamiento
train_generator_letras = datagen_letras.flow_from_directory(
    train_dir_letras,
    target_size=(28, 28),  # Redimensionar las imágenes a 28x28
    color_mode='grayscale',  # Convertir las imágenes a escala de grises
    class_mode='categorical',  # Clasificación categórica (para letras A-Z)
    batch_size=32
)

# Crear el modelo CNN para letras
def crear_modelo_letras(input_shape, num_clases):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_clases, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Crear el modelo para letras (input_shape es 28x28x1 porque las imágenes son en escala de grises)
input_shape = (28, 28, 1)
num_clases = len(train_generator_letras.class_indices)  # Detectar automáticamente cuántas clases hay (A-Z)
modelo_letras = crear_modelo_letras(input_shape, num_clases)

# Entrenar el modelo
modelo_letras.fit(train_generator_letras, epochs=50)

# Guardar el modelo entrenado
modelo_letras.save('modelo_detectar_letras.h5')

# Cargar el modelo previamente entrenado
modelo_letras = load_model('modelo_detectar_letras.h5')

# Definir la ruta de la carpeta de segmentación de matrículas
segmentacion_dir = 'C:/Users/eduar/OneDrive/Escritorio/Matriculas/segmentacion_matriculas'  # Cambia esto por tu ruta

# Definir el tamaño de las imágenes que el modelo espera (28x28 en escala de grises)
image_size = (28, 28)

# Ruta del archivo CSV con las matrículas
csv_file = 'matriculas.csv'  # Cambia esto por la ruta real del archivo CSV

# Leer el archivo CSV y verificar las columnas
matriculas_df = pd.read_csv(csv_file, delimiter=';')  # Cambié el delimitador por ';', que parece ser el correcto para tu archivo

# Imprimir las columnas del CSV para verificar los nombres correctos
print(matriculas_df.columns)

# Crear un diccionario de nombres de imágenes a matrículas
imagenes_matriculas = dict(zip(matriculas_df['Imagen'], matriculas_df['Matricula']))

# Obtener las carpetas dentro de la carpeta de segmentación de matrículas
subcarpetas = [os.path.join(segmentacion_dir, d) for d in os.listdir(segmentacion_dir) if os.path.isdir(os.path.join(segmentacion_dir, d))]

# Cargar las 3 últimas imágenes de cada subcarpeta y hacer predicciones
for carpeta in subcarpetas:
    # Obtener el nombre de la subcarpeta (por ejemplo, "caracter_5")
    nombre_subcarpeta = os.path.basename(carpeta)
    
    # Mostrar solo el nombre de la carpeta (la parte final de la ruta)
    print(f"Predicciones para las últimas 3 imágenes de la carpeta: {nombre_subcarpeta}")
    
    # Obtener la lista de archivos de imágenes dentro de la carpeta
    imagenes = [f for f in os.listdir(carpeta) if f.endswith('.png') or f.endswith('.jpg')]  # Asegúrate de que las imágenes tengan estas extensiones
    
    # Ordenar las imágenes y tomar las últimas 3
    imagenes_ultimas = sorted(imagenes)[-3:]
    
    # Listar las letras predichas para las 3 últimas imágenes
    letras_predichas = []
    
    for img_name in imagenes_ultimas:
        img_path = os.path.join(carpeta, img_name)
        
        # Cargar la imagen
        img = image.load_img(img_path, target_size=image_size, color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expandir la dimensión para que coincida con la entrada del modelo (1, 28, 28, 1)
        img_array /= 255.0  # Normalizar los valores de los píxeles
        
        # Hacer la predicción
        prediccion = modelo_letras.predict(img_array)
        
        # Obtener el índice de la clase con la probabilidad más alta
        clase_predicha = np.argmax(prediccion, axis=1)[0]
        
        # Obtener la etiqueta correspondiente a esa clase
        letra_predicha = list(train_generator_letras.class_indices.keys())[clase_predicha]
        
        letras_predichas.append(letra_predicha)
    
    # Concatenar las 3 letras predichas
    letras_predichas_juntas = ''.join(letras_predichas)
    
    # Imprimir la combinación de las 3 letras predichas
    print(f"Predicción combinada para las 3 últimas imágenes: {letras_predichas_juntas}")
    
    # Buscar la matrícula en el CSV usando el nombre de la subcarpeta (nombre_subcarpeta)
    matricula = imagenes_matriculas.get(nombre_subcarpeta)
    
    if matricula:
        matricula_letras = matricula[-3:]  # Extraer las 3 últimas letras
        
        # Comparar las letras predichas con las letras reales
        if letras_predichas_juntas == matricula_letras:
            print(f"¡Correcto! Las letras predichas coinciden con la matrícula: {matricula}")
        else:
            print(f"¡Incorrecto! Las letras predichas: {letras_predichas_juntas}, Matrícula real: {matricula}")
    else:
        print(f"No se encontró matrícula para la subcarpeta {nombre_subcarpeta}.")


