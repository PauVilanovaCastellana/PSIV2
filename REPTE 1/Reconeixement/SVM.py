import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Ruta de la carpeta 'caracters'
base_path = 'Elements'

# Listas para almacenar imágenes y etiquetas
X = []
y = []

# Cargar imágenes y etiquetas
for label in os.listdir(base_path):
    label_path = os.path.join(base_path, label)
    if os.path.isdir(label_path):  # Asegurarse de que es un directorio
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            # Leer y procesar la imagen
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
            img = cv2.resize(img, (28, 28))  # Redimensionar a un tamaño uniforme
            X.append(img.flatten())  # Aplanar la imagen y añadir a la lista
            y.append(label)  # Añadir la etiqueta

# Convertir listas a matrices de NumPy
X = np.array(X)
y = np.array(y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el clasificador SVM
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Realizar predicciones
y_pred = clf.predict(X_test)

# Imprimir el informe de clasificación
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(clf, 'svm_model.pkl')

