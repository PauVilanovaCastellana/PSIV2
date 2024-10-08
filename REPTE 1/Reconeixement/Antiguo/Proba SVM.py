import cv2
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model


# Cargar los modelos SVM
letter_svm = joblib.load('letter_svm_classifier_model.pkl')
number_svm = joblib.load('number_svm_classifier_model.pkl')

# Cargar los modelos de red neuronal para extraer características
letter_feature_extractor = load_model('letter_feature_extractor_model.keras')
number_feature_extractor = load_model('number_feature_extractor_model.keras')

# Función para preprocesar la imagen
def preprocess_image(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Redimensionar a 28x28
    img = cv2.resize(img, (28, 28))
    # Ajustar el formato para la red neuronal
    reshaped_img = img.reshape(1, 28, 28, 1)  # Para la entrada de la red neuronal
    # Aplanar la imagen para el clasificador
    flattened_img = img.flatten()
    return flattened_img, reshaped_img

# Función para clasificar imágenes de letras o números
def classify_image(image_path, is_letter):
    # Preprocesar la imagen
    flattened_img, reshaped_img = preprocess_image(image_path)
    
    if is_letter:
        # Usar la red neuronal para extraer características de la letra
        letter_features = letter_feature_extractor.predict(reshaped_img)
        # Usar el SVM de letras para la clasificación final
        predicted_label = letter_svm.predict(letter_features)[0]
        return f'Letra: {predicted_label}'
    else:
        # Usar la red neuronal para extraer características del número
        number_features = number_feature_extractor.predict(reshaped_img)
        # Usar el SVM de números para la clasificación final
        predicted_label = number_svm.predict(number_features)[0]
        return f'Número: {predicted_label}'

# Ruta de la carpeta de imágenes a probar
test_image_path = 'segmentacion_matriculas/1'  # Cambia esto por la ruta de tu carpeta

# Recorrer todas las imágenes en la carpeta de prueba
image_files = [filename for filename in os.listdir(test_image_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]

# Clasificar las primeras 4 imágenes como números y las 3 restantes como letras
for idx, filename in enumerate(image_files):
    image_path = os.path.join(test_image_path, filename)
    
    # Decidir si es letra o número
    is_letter = idx >= 4  # Las imágenes con índice 4 o más son letras

    # Clasificar la imagen
    result = classify_image(image_path, is_letter)

    # Imprimir el resultado
    print(f'La imagen {filename} fue clasificada como: {result}')
