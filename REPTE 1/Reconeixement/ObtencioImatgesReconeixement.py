import cv2
import numpy as np
import os

# Ruta de la fuente Matricula Española
fuente_ruta = "matricula_espanola.ttf"

# Texto que queremos separar en imágenes
texto = "1234ABCDEFGHIJKLMNOPQRSTUVWXYZ"   

# Ángulos de rotación
angulos_rotacion = [-9, -6, -3, 0, 3, 6, 9]

# Crear directorio para guardar imágenes
directorio = "caracters"
if not os.path.exists(directorio):
    os.makedirs(directorio)

# Recorrer cada carácter del texto
for i, caracter in enumerate(texto):
    # Crear una imagen negra para el carácter
    imagen = np.zeros((150, 150, 3), dtype=np.uint8)  # Imagen negra

    # Establecer la posición del texto
    text_size = cv2.getTextSize(caracter, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    text_x = (imagen.shape[1] - text_size[0]) // 2
    text_y = (imagen.shape[0] + text_size[1]) // 2

    # Dibujar el carácter en la imagen
    cv2.putText(imagen, caracter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Crear carpeta para el carácter
    caracter_dir = os.path.join(directorio, caracter)
    if not os.path.exists(caracter_dir):
        os.makedirs(caracter_dir)

    # Aplicar rotaciones
    for angulo in angulos_rotacion:
        # Obtener la matriz de rotación
        h, w = imagen.shape[:2]
        centro = (w // 2, h // 2)
        matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

        # Rotar la imagen
        imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (w, h))

        # Convertir a escala de grises y aplicar umbral
        gris_rotado = cv2.cvtColor(imagen_rotada, cv2.COLOR_BGR2GRAY)
        _, umbral_rotado = cv2.threshold(gris_rotado, 240, 255, cv2.THRESH_BINARY_INV)

        # Encontrar contornos
        contornos, _ = cv2.findContours(umbral_rotado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            # Encontrar el contorno más grande
            c = max(contornos, key=cv2.contourArea)
            # Obtener el contorno como una máscara
            mask = np.zeros_like(imagen_rotada)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Recortar la imagen utilizando la máscara
            imagen_recortada = cv2.bitwise_and(imagen_rotada, mask)

            # Convertir a escala de grises para ajustar el tamaño
            gris_recortado = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2GRAY)
            _, umbral_recortado = cv2.threshold(gris_recortado, 1, 255, cv2.THRESH_BINARY)

            # Encontrar los contornos de la imagen recortada
            contornos_recortados, _ = cv2.findContours(umbral_recortado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contornos_recortados:
                c_recortado = max(contornos_recortados, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c_recortado)

                # Recortar la imagen en función del contorno
                imagen_final = imagen_recortada[y:y + h, x:x + w]

                # Guardar la imagen final en un archivo
                img_path = f"{caracter}_{angulo}.png"
                img_output_path = os.path.join(caracter_dir, img_path)
                cv2.imwrite(img_output_path, imagen_final)

                # Mostrar la imagen (opcional)
                cv2.imshow(f'Caracter: {caracter} - Angulo: {angulo}', imagen_final)
                cv2.waitKey(100)  # Esperar un corto tiempo para ver la imagen

cv2.destroyAllWindows()
