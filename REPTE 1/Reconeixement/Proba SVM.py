import cv2
import numpy as np
import joblib

# Cargar el modelo SVM
model = joblib.load('svm_model.pkl')

# Función para preprocesar la imagen de entrada
def preprocess_image(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar a 28x28
    img = cv2.resize(img, (28, 28))
    # Aplanar la imagen
    img = img.flatten()
    return img

# Ruta de la imagen a probar
test_image_path = 'segmentacion_matriculas/1/caracter_7.png'  # Cambia esto por la ruta de tu imagen

# Preprocesar la imagen
test_image = preprocess_image(test_image_path)

# Realizar la predicción
predicted_label = model.predict([test_image])

# Imprimir el resultado
print(f'La imagen fue clasificada como: {predicted_label[0]}')
