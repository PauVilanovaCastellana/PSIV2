import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Función para cargar imágenes y preprocesarlas
def preprocess_image(img_path, img_width, img_height):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)

# Función para predecir números
def predict_numbers(model, images):
    predictions = model.predict(np.array(images))
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels

# Función para predecir letras
def predict_letters(model, images):
    images = np.array(images).reshape(-1, 30, 30, 1)  # Preprocesar imágenes para la red de letras
    features = model.predict(images)  # Extraer características con la red de letras
    return np.argmax(features, axis=1)  # Predicción final con la SVM


