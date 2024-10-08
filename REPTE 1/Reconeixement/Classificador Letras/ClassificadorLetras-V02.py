import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# Establecer la semilla para reproducibilidad
SEED = 41

# Para TensorFlow
tf.random.set_seed(SEED)

# Para NumPy
np.random.seed(SEED)

# Para cualquier otro código que dependa de la aleatoriedad, como el SVM
random.seed(SEED)

# 1. Cargar solo imágenes de letras
def load_letter_data(directory):
    images = []
    labels = []
    for folder_name in os.listdir(directory):
        # Filtramos solo las carpetas con letras
        if folder_name.isalpha():  
            folder_path = os.path.join(directory, folder_name)
            label = folder_name  # La etiqueta será la letra misma
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (30, 30))  # Redimensionamos las imágenes a 28x28
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# 2. Definir la red neuronal para la extracción de características
def create_feature_extractor():
    model = models.Sequential([
        layers.Input(shape=(30, 30, 1)),  # Imágenes en escala de grises
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),  # Aplanamos para obtener un vector de características
        layers.Dense(128, activation='relu')  # Capa densa para la extracción de características
    ])
    return model

# 3. Extraer características de las imágenes
def extract_features(model, images):
    images = images.reshape(-1, 30, 30, 1)  # Aseguramos el formato correcto
    return model.predict(images)

# 4. Clasificador SVM
def train_svm(features, labels):
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(features, labels)
    return svm_classifier

# Guardar el modelo de la red neuronal
def save_neural_network(model, path):
    model.save(path)

# Guardar el modelo SVM
def save_svm_model(svm_classifier, path):
    joblib.dump(svm_classifier, path)

# Cargar los datos de letras
images, labels = load_letter_data('elements')

# Crear y entrenar la red neuronal para extraer características
feature_extractor = create_feature_extractor()
features = extract_features(feature_extractor, images)

# Entrenar la SVM con todas las características extraídas
svm_classifier = train_svm(features, labels)

# Guardar los modelos
save_neural_network(feature_extractor, 'letter_feature_extractor_model.keras')  # Guardamos la red neuronal
save_svm_model(svm_classifier, 'letter_svm_classifier_model.keras')               # Guardamos el modelo SVM

print('Clasificador de letras entrenado y modelos guardados con éxito.')
