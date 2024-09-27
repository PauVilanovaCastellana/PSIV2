import cv2
import numpy as np
import os

def recortar_caracteres(imagen_umbral):
    # Aplicar el detector de bordes Canny para ayudar a detectar contornos
    imagen_canny = cv2.Canny(imagen_umbral, 100, 200)

    # Aplicar dilatación para unir caracteres antes de encontrar contornos
    kernel = np.ones((3, 3), np.uint8)
    imagen_dilatada = cv2.dilate(imagen_canny, kernel, iterations=2)

    # Encontrar contornos en la imagen dilatada (después de Canny)
    contornos, _ = cv2.findContours(imagen_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las imágenes recortadas de los caracteres
    caracteres_recortados = []

    # Obtener dimensiones de la imagen original umbralizada
    altura_img, anchura_img = imagen_umbral.shape[:2]

    for contorno in contornos:
        # Obtener el rectángulo delimitador de cada contorno
        x, y, w, h = cv2.boundingRect(contorno)

        # Filtrar los contornos que están demasiado cerca de los bordes de la imagen
        margen_borde = 2  # Ajusta este valor según el margen que quieras permitir

        if (x > margen_borde and y > margen_borde and
            x + w < anchura_img - margen_borde and
            y + h < altura_img - margen_borde):
            
            # Filtrar por altura mínima y proporción de ancho/alto
            if h > 50 and 0.1 < w / h < 1:  # Proporciones típicas de caracteres
                # Dibujar la bounding box (caja delimitadora) en la imagen umbralizada
                cv2.rectangle(imagen_umbral, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Color rojo y grosor 2

                # Recortar la imagen original (imagen_umbral) usando el rectángulo delimitador
                caracter_recortado = imagen_umbral[y:y+h, x:x+w]
                
                # Añadir la imagen recortada a la lista (junto con la coordenada x para ordenar)
                caracteres_recortados.append((x, caracter_recortado))

    # Devolver las imágenes de los caracteres junto con las coordenadas X
    return caracteres_recortados


def ordenar_caracteres_por_x(caracteres_recortados):
    """
    Ordena los caracteres recortados según su posición en X (de izquierda a derecha).
    """
    # Ordenar los caracteres por su coordenada X
    caracteres_ordenados = sorted(caracteres_recortados, key=lambda x: x[0])
    
    # Devolver solo las imágenes ordenadas
    return [caracter[1] for caracter in caracteres_ordenados]


def load_image(img_path):
    """
    Carga una imagen desde la ruta proporcionada.
    """
    return cv2.imread(img_path)


def mostrar_imagen_con_caracteres(imagen, caracteres_recortados):
    """
    Muestra la imagen con los caracteres resaltados.
    """
    cv2.imshow('Imagen con caracteres resaltados', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def procesar_imagen(img_path, output_dir):
    """
    Procesa una única imagen para extraer los caracteres y guardarlos en la carpeta de salida.
    """
    img = load_image(img_path)
    
    # Ajustar el brillo de la imagen aumentada
    brillo_alpha = 1.46  # Factor de contraste
    brillo_beta = 70  # Valor de brillo
    
    imagen_brillo = cv2.convertScaleAbs(img, alpha=brillo_alpha, beta=brillo_beta)
    
    # Convertir a escala de grises y luego a HSV
    imagen_hsv = cv2.cvtColor(imagen_brillo, cv2.COLOR_BGR2HSV)
    canal_v = imagen_hsv[:, :, 2]  # Usamos el canal de valor (V) para umbralizar
    
    # Aplicar umbral inverso
    _, imagen_umbral = cv2.threshold(canal_v, 220, 255, cv2.THRESH_BINARY_INV)

    caracteres_recortados=recortar_caracteres(imagen_umbral)
    # Ordenar los caracteres recortados por la coordenada X
    caracteres_ordenados = ordenar_caracteres_por_x(caracteres_recortados)

    # Guardar los caracteres recortados en la carpeta de salida
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    carpeta_salida = os.path.join(output_dir, base_filename)
    
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Guardar cada caracter en la carpeta
    for i, caracter in enumerate(caracteres_ordenados):
        caracter_filename = os.path.join(carpeta_salida, f"caracter_{i+1}.png")
        cv2.imwrite(caracter_filename, caracter)
        print(f"Guardado: {caracter_filename}")

    # Mostrar la imagen con los caracteres resaltados
    mostrar_imagen_con_caracteres(imagen_umbral, caracteres_recortados)


def procesar_directorio(input_dir, output_dir):
    """
    Procesa todas las imágenes de un directorio y guarda los caracteres segmentados.
    """
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Recorrer todas las imágenes del directorio
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            print(f"Procesando: {img_path}")
            procesar_imagen(img_path, output_dir)


if __name__ == "__main__":
    input_dir = "imatges_matriculas"  # Carpeta con las imágenes de matrículas
    output_dir = "segmentacion_matriculas"  # Carpeta donde se guardarán los caracteres segmentados
    
    procesar_directorio(input_dir, output_dir)
