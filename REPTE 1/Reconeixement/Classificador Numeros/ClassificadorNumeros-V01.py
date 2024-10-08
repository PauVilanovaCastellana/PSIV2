import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Tamaño de las imágenes actualizado a 33x47
img_width = 33
img_height = 47

# Función para cargar imágenes y etiquetas de las carpetas de dígitos
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label in range(10):  # Asumiendo que solo hay carpetas para dígitos 0-9
        digit_folder = os.path.join(folder_path, str(label))
        if os.path.exists(digit_folder):
            for filename in os.listdir(digit_folder):
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    img_path = os.path.join(digit_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_width, img_height))  # Redimensionar a 33x47
                    img = img.astype('float32') / 255.0
                    images.append(np.expand_dims(img, axis=-1))
                    labels.append(label)
    return np.array(images), np.array(labels)

# Cargar las imágenes y las etiquetas
folder_path = 'caracters'  # Ruta de la carpeta principal donde están las carpetas de los dígitos
images, labels = load_images_from_folder(folder_path)

# Convertir etiquetas a formato categórico (one-hot encoding)
labels = to_categorical(labels, num_classes=10)

# Dividir en conjunto de entrenamiento y validación (80% entrenamiento, 20% validación)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Crear un generador de imágenes con aumentación (opcional)
datagen = ImageDataGenerator(
    rotation_range=10,     # Rotar imágenes hasta 10 grados
    width_shift_range=0.1, # Mover imágenes horizontalmente
    height_shift_range=0.1, # Mover imágenes verticalmente
    zoom_range=0.1         # Zoom dentro de la imagen
)

# Ajustar el generador con las imágenes de entrenamiento
datagen.fit(X_train)

# Definir el modelo con Dropout y regularización L2
model = Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))  # Regularización L2
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    validation_data=(X_val, y_val), 
                    epochs=50)

# Guardar el modelo
model.save('clasificador_numeros_33x47.keras')

# Gráfica de precisión (accuracy)
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica de pérdida (loss)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
