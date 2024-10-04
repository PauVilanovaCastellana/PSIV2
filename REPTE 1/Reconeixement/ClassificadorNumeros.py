import os
import cv2
import numpy as np
import joblib
from sklearn import svm
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar solo imágenes de números
def load_number_data(directory):
    images = []
    labels = []
    for folder_name in os.listdir(directory):
        # Filtramos solo las carpetas con números
        if folder_name.isdigit():  
            folder_path = os.path.join(directory, folder_name)
            label = folder_name  # La etiqueta será el número mismo
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))  # Redimensionamos las imágenes a 28x28
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# 2. Definir la red neuronal para la extracción de características
def create_feature_extractor():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Imágenes en escala de grises
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
    images = images.reshape(-1, 28, 28, 1)  # Aseguramos el formato correcto
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

# Cargar los datos de números
images, labels = load_number_data('caracters')

# Crear y entrenar la red neuronal para extraer características
feature_extractor = create_feature_extractor()
features = extract_features(feature_extractor, images)

# Entrenar la SVM con todas las características extraídas
svm_classifier = train_svm(features, labels)

# Guardar los modelos
save_neural_network(feature_extractor, 'number_feature_extractor_model.keras')  # Guardamos la red neuronal
save_svm_model(svm_classifier, 'number_svm_classifier_model.pkl')               # Guardamos el modelo SVM

print('Clasificador de números entrenado y modelos guardados con éxito.')
