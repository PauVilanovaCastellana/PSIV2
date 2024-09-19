import cv2
import numpy as np


def recortar_caracteres(imagen_umbral):
    # Obtener dimensiones de la imagen
    altura_img, anchura_img = imagen_umbral.shape[:2]

    # Encontrar contornos en la imagen umbral
    contornos, _ = cv2.findContours(imagen_umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las imágenes recortadas de los caracteres
    caracteres_recortados = []

    for contorno in contornos:
        # Obtener el rectángulo delimitador de cada contorno
        x, y, w, h = cv2.boundingRect(contorno)

        # Filtrar los contornos que están demasiado cerca de los bordes de la imagen
        margen_borde = 2  # Puedes ajustar este valor según el margen que quieras permitir

        if (x > margen_borde and y > margen_borde and
            x + w < anchura_img - margen_borde and
            y + h < altura_img - margen_borde):
            
            # Filtrar por altura mínima y proporción de ancho/alto
            if h > 50 and 0.2 < w / h < 0.9:  # Proporciones típicas de caracteres
                # Dibujar la bounding box (caja delimitadora) en la imagen umbralizada
                cv2.rectangle(imagen_umbral, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Color blanco y grosor 2
                
                # Recortar la imagen usando el rectángulo delimitador
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


def main():
    # Configuración
    img_path = "imatges_matriculas/12.jpeg"
    
    img = load_image(img_path)
    
    # Ajustar el brillo de la imagen aumentada
    brillo_alpha = 1.45  # Factor de contraste
    brillo_beta = 90  # Valor de brillo (puedes ajustar este valor)
    imagen_brillo = cv2.convertScaleAbs(img, alpha=brillo_alpha, beta=brillo_beta)
    
    # Mostrar la imagen con el brillo ajustado
    cv2.imshow('Imagen Brillo Ajustado', imagen_brillo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convertir a escala de grises y luego a HSV
    imagen_hsv = cv2.cvtColor(imagen_brillo, cv2.COLOR_BGR2HSV)
    canal_v = imagen_hsv[:, :, 2]  # Usamos el canal de valor (V) para umbralizar
    
    # Aplicar umbral inverso
    _, imagen_umbral = cv2.threshold(canal_v, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Recortar los caracteres de la imagen (y dibujar las bounding boxes)
    caracteres = recortar_caracteres(imagen_umbral)
    
    # Mostrar la imagen umbralizada con las bounding boxes dibujadas
    cv2.imshow('Imagen Umbralizada con Bounding Boxes', imagen_umbral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Ordenar los caracteres recortados por la coordenada X
    caracteres_ordenados = ordenar_caracteres_por_x(caracteres)
    
    # Mostrar las imágenes recortadas de los caracteres (en orden)
    for i, caracter in enumerate(caracteres_ordenados):
        cv2.imshow(f'Caracter {i}', caracter)
        
        # Esperar a que se presione una tecla para mostrar el siguiente carácter
        print(f"Mostrando carácter {i + 1} de {len(caracteres_ordenados)}")
        cv2.waitKey(0)  # Esperar a que se presione una tecla antes de mostrar el siguiente
    
    # Cerrar todas las ventanas al final
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


