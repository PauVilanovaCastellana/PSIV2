import os
import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

# Función para calcular el porcentaje de acierto
def calculate_accuracy(predictions, actual):
    correct = sum([p == a for p, a in zip(predictions, actual)])
    return (correct / len(actual)) * 100

# Cargar los modelos guardados
number_model = load_model('clasificador_numeros_33x47.keras')  # Modelo para los números
letter_feature_extractor = load_model('letter_feature_extractor_model.keras')  # Red de extracción de características para letras
letter_svm = joblib.load('letter_svm_classifier_model.keras')  # Clasificador SVM para letras

# Leer el archivo CSV con las imágenes y matrículas
df = pd.read_csv('matriculas.csv', sep=';')

# Definir las letras permitidas
allowed_letters = list("ABCDEFGHJKLMNPRSTVWXYZ")

# Listas para almacenar los resultados de predicción, etiquetas reales y el % de aciertos
predicciones_totales = []
aciertos_numeros = []
aciertos_letras = []
true_numbers = []
true_letters = []
predicted_numbers_total = []
predicted_letters_total = []

# Recorrer cada fila del CSV
for index, row in df.iterrows():
    img_folder = f"segmentacion_matriculas/{row['Imagen']}"  # Ruta de la carpeta de imágenes
    matricula_real = row['Matricula']

    # Cargar y preprocesar las primeras 4 imágenes para predecir los números
    number_images = []
    for img_name in sorted(os.listdir(img_folder))[:4]:  # Las primeras 4 imágenes
        img_path = os.path.join(img_folder, img_name)
        number_images.append(preprocess_image(img_path, 33, 47))  # Preprocesar para el modelo de números

    predicted_numbers = predict_numbers(number_model, number_images)
    predicted_number_str = ''.join(map(str, predicted_numbers))  # Convertir las predicciones a string

    # Cargar y preprocesar las últimas 3 imágenes para predecir las letras
    letter_images = []
    for img_name in sorted(os.listdir(img_folder))[-3:]:  # Las últimas 3 imágenes
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 30))  # Preprocesar para el modelo de letras
        letter_images.append(img)

    predicted_letter_features = letter_feature_extractor.predict(np.array(letter_images).reshape(-1, 30, 30, 1))
    predicted_letters = letter_svm.predict(predicted_letter_features)
    predicted_letter_str = ''.join(predicted_letters)

    # Predicción final para la matrícula completa
    prediccion_final = predicted_number_str + predicted_letter_str
    predicciones_totales.append(prediccion_final)

    # Guardar los números y letras reales y predichos
    true_numbers.extend(list(map(int, matricula_real[:4])))  # Números reales
    true_letters.extend(list(matricula_real[4:]))  # Letras reales
    predicted_numbers_total.extend(predicted_numbers)  # Números predichos
    predicted_letters_total.extend(predicted_letters)  # Letras predichas

    # Calcular el % de acierto para los números y las letras
    accuracy_numbers = calculate_accuracy(predicted_numbers, list(map(int, matricula_real[:4])))  # Comparar los números
    accuracy_letters = calculate_accuracy(predicted_letters, list(matricula_real[4:]))  # Comparar las letras

    # Guardar los porcentajes de acierto
    aciertos_numeros.append(accuracy_numbers)
    aciertos_letras.append(accuracy_letters)

# Añadir las predicciones y los porcentajes de acierto al DataFrame
df['Prediccion'] = predicciones_totales
df['Acierto_Numeros'] = aciertos_numeros
df['Acierto_Letras'] = aciertos_letras

# Guardar el DataFrame actualizado en un nuevo CSV
df.to_csv('resultados_predicciones.csv', index=False)

print('Predicciones realizadas y guardadas con éxito en resultados_predicciones.csv.')

# Matriz de confusión para números
cm_numbers = confusion_matrix(true_numbers, predicted_numbers_total)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_numbers, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Numbers')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Reporte de clasificación para números
print('Classification Report for Numbers:')
print(classification_report(true_numbers, predicted_numbers_total))

# Matriz de confusión para letras (solo usando las letras permitidas)
cm_letters = confusion_matrix(true_letters, predicted_letters_total, labels=allowed_letters)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_letters, annot=True, fmt='d', cmap='Blues', xticklabels=allowed_letters, yticklabels=allowed_letters)
plt.title('Confusion Matrix for Letters')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Reporte de clasificación para letras
print('Classification Report for Letters:')
print(classification_report(true_letters, predicted_letters_total, labels=allowed_letters))

