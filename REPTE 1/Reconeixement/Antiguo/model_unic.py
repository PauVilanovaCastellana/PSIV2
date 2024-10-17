import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from PIL import Image

# Descomprimir el archivo Elements.zip
zip_path = "Elements.zip"
extract_dir = "elements"

# Cargar VGG16 y crear el extractor de características
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Definir tamaño de imagen
image_size = (224, 224)

# Aumento de datos
data_augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Listas para almacenar datos y etiquetas del conjunto de entrenamiento
data = []
labels = []

# Leer cada carpeta correspondiente a un carácter en Elements.zip
for character_folder in os.listdir(extract_dir):
    folder_path = os.path.join(extract_dir, character_folder)

    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)

            # Leer la imagen en escala de grises
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Error al cargar la imagen {img_path}.")
                continue

            # Redimensionar la imagen a 224x224 (como lo espera VGG16)
            img_resized = cv2.resize(img, image_size)

            # Convertir a RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

            # Preprocesar la imagen
            img_preprocessed = preprocess_input(img_rgb)
            img_expanded = np.expand_dims(img_preprocessed, axis=0)

            # Extraer características usando VGG16
            features = feature_extractor.predict(img_expanded)
            features_flattened = features.flatten()

            # Almacenar las características y las etiquetas
            data.append(features_flattened)
            labels.append(character_folder)

# Convertir a numpy arrays
X = np.array(data)
y = np.array(labels)

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)  # One-hot encoding

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear un modelo simple con capa completamente conectada sobre las características extraídas por VGG16
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Softmax para clasificación

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---- ENTRENAR EL MODELO Y GUARDAR LA HISTORIA ----
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ---- CURVA DE ACCURACY A LO LARGO DE LAS ÉPOCAS ----
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ---- MATRIZ DE CONFUSIÓN ----
# Predicciones sobre el conjunto de prueba
y_test_pred = model.predict(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
y_test_true_labels = np.argmax(y_test, axis=1)

# Generar la matriz de confusión
cm = confusion_matrix(y_test_true_labels, y_test_pred_labels)

# Mostrar la matriz de confusión gráficamente
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)  # Rotar las etiquetas para que se vean mejor
plt.show()



