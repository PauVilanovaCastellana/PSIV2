import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import random

# Establecer la semilla para reproducibilidad
SEED = 26
np.random.seed(SEED)
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
                img = cv2.resize(img, (30, 30))  # Redimensionamos las imágenes a 30x30
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# 2. Extraer características aplanando las imágenes
def extract_features(images):
    features = []
    for img in images:
        img_flattened = img.flatten()  # Aplanar la imagen a un vector 1D
        features.append(img_flattened)
    return np.array(features)

# 3. Clasificador SVM
def train_svm(features, labels):
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(features, labels)
    return svm_classifier

# 4. Validación cruzada para el SVM
def cross_validate_svm(features, labels, cv_folds=5):
    svm_classifier = svm.SVC(kernel='linear')  # SVM con kernel lineal
    scores = cross_val_score(svm_classifier, features, labels, cv=cv_folds)
    return scores

# Guardar el modelo SVM
def save_svm_model(svm_classifier, path):
    joblib.dump(svm_classifier, path)

# Cargar los datos de letras
images, labels = load_letter_data('elements')

# Extraer características aplanadas de las imágenes
features = extract_features(images)

# Entrenar la SVM con todas las características extraídas
svm_classifier = train_svm(features, labels)

# Guardar el modelo SVM
save_svm_model(svm_classifier, 'letter_svm_classifier_model.pkl')  # Guardamos el modelo SVM

print('Clasificador de letras entrenado y modelo SVM guardado con éxito.')

# Aplicar validación cruzada al clasificador SVM
cv_scores = cross_validate_svm(features, labels, cv_folds=5)
print(f"Puntuaciones de validación cruzada: {cv_scores}")
print(f"Media de las puntuaciones: {np.mean(cv_scores)}")

# Reporte de clasificación
predictions = svm_classifier.predict(features)
print(classification_report(labels, predictions))
