import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def recortar_caracteres(imagen_umbral):
    # Encontrar contornos en la imagen umbral
    contornos, _ = cv2.findContours(imagen_umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las imágenes recortadas de los caracteres
    caracteres_recortados = []

    for contorno in contornos:
        # Obtener el rectángulo delimitador de cada contorno
        x, y, w, h = cv2.boundingRect(contorno)
        
        if h > 50 and w > 30 and x != 0 and y != 0:
            # Recortar la imagen usando el rectángulo delimitador
            caracter_recortado = imagen_umbral[y:y+h, x:x+w]
    
            # Añadir la imagen recortada a la lista
            caracteres_recortados.append(caracter_recortado)

    return caracteres_recortados

def load_image(img_path):
    """
    Carga una imagen desde la ruta proporcionada.
    """
    return cv2.imread(img_path)


def main():
    # Configuración
    img_path = "imatges_matriculas/4.jpeg"
    
    img = load_image(img_path)
    
    # Ajustar el brillo de la imagen aumentada
    brillo_alpha = 2 # Factor de contraste
    brillo_beta = 35   # Valor de brillo (puedes ajustar este valor)
    imagen_brillo = cv2.convertScaleAbs(img, alpha=brillo_alpha, beta=brillo_beta)
    
    cv2.imshow('Imagen Digitos', imagen_brillo)
    
    # Esperar a que se presione una tecla para cerrar la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convertir a escala de grises
    imagen_hsv = cv2.cvtColor(imagen_brillo, cv2.COLOR_BGR2HSV)
    canal_v = imagen_hsv[:, :, 2]
    
    # Aplicar umbral
    _, imagen_umbral = cv2.threshold(canal_v, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Mostrar la imagen umbralizada
    cv2.imshow('Imagen Umbral', imagen_umbral)
    
    # Esperar a que se presione una tecla para cerrar la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Mostrar la imagen redimensionada
    cv2.imshow('Imagen Umbral Redimensionada', imagen_umbral)
    
    # Esperar a que se presione una tecla para cerrar la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Uso de la función recortar_caracteres
    caracteres = recortar_caracteres(imagen_umbral)
    
    # Mostrar las imágenes recortadas de caracteres
    for i, caracter in enumerate(caracteres):
        cv2.imshow(f'Caracter {i}', caracter)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()