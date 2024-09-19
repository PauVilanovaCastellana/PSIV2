# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:06:06 2024

@author: jaaa2
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_model(model_path):
    """
    Carga el modelo YOLO desde la ruta proporcionada.
    """
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def load_image(img_path):
    """
    Carga una imagen desde la ruta proporcionada.
    """
    return cv2.imread(img_path)

def detect_objects(model, img):
    """
    Realiza la detección de objetos en la imagen utilizando el modelo.
    """
    results = model(img)
    return results.xyxy[0].cpu().numpy()

def get_best_detection(detections):
    """
    Obtiene la detección con mayor confianza de entre las detecciones proporcionadas.
    """
    if len(detections) > 0:
        return max(detections, key=lambda x: x[4])
    return None

def draw_detection(img, detection):
    """
    Dibuja el recuadro y el texto "Matrícula" sobre la imagen para la detección proporcionada.
    Además, devuelve las coordenadas del recuadro.
    """
    x1, y1, x2, y2, _, _ = map(int, detection)
    
    # Dibujar el recuadro en la imagen OpenCV
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Convertir la imagen de OpenCV (BGR) a PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Cargar la fuente de PIL
    font_path = "C:/Windows/Fonts/arial.ttf"  # Usa una fuente válida en tu sistema
    font_size = 36
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: No se puede abrir la fuente en la ruta {font_path}.")
        font = ImageFont.load_default()
    
    # Calcular el tamaño del texto
    text = 'Matrícula'
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calcular la posición para centrar el texto
    text_x = x1 + (x2 - x1 - text_width) / 2
    text_y = y1 - text_height - 10
    
    # Dibujar el texto centrado en la imagen PIL
    draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))
    
    # Convertir de nuevo la imagen PIL a OpenCV (BGR)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img, (x1, y1, x2, y2)

def recortar_imagen_con_coordenadas(image, coords):
    """
    Recorta la imagen usando las coordenadas proporcionadas.
    """
    x1, y1, x2, y2 = coords
    # Recortar la imagen usando slicing en NumPy
    recorte = image[y1:y2, x1:x2]
    return recorte

def extraer_canal_v(img_crop):
    """
    Convierte una imagen al espacio HSV, extrae solo el canal V y muestra la imagen resultante.
    """
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Extraer el canal V
    canal_v = hsv[:, :, 2]
    
    # Mostrar el canal V
    cv2.imshow('Canal V', canal_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return canal_v

def procesar_y_enderezar_letras(img_crop):
    """
    Convierte la imagen a HSV, extrae el canal V, aplica el detector de bordes Canny,
    y detecta los contornos que correspondan a las letras o números dentro de la matrícula.
    Luego, rota la imagen para que las letras queden horizontales (0 grados).
    """
    if img_crop is None or img_crop.size == 0:
        raise ValueError("La imagen no se ha cargado correctamente o está vacía.")
    
    # Convertir la imagen a HSV
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Extraer el canal V (valor)
    canal_v = hsv[:, :, 2]
    
    # Aplicar el detector de bordes Canny
    bordes_canny = cv2.Canny(canal_v, 50, 150)
    
    # Encontrar contornos en la imagen de bordes
    contornos, _ = cv2.findContours(bordes_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        raise ValueError("No se detectaron contornos en la imagen.")
    
    # Filtrar los contornos para encontrar los que correspondan a letras/números
    contornos_letras = []
    for contorno in contornos:
        # Filtrar por tamaño mínimo de contorno (para evitar ruidos pequeños)
        area = cv2.contourArea(contorno)
        if area > 100:  # Este valor se puede ajustar dependiendo del tamaño esperado de las letras
            contornos_letras.append(contorno)
    
    if not contornos_letras:
        raise ValueError("No se encontraron contornos significativos que representen letras o números.")
    
    # Encontrar la caja delimitadora mínima que contenga todas las letras/números
    contornos_todos = np.vstack(contornos_letras)
    rect = cv2.minAreaRect(contornos_todos)
    angulo = rect[2]  # Este es el ángulo de inclinación del rectángulo que contiene las letras
    
    # Si el ángulo es negativo (hacia la izquierda), ajusta el ángulo correctamente
    if angulo < -45:
        angulo = 90 + angulo
    elif angulo > 45:
        angulo = angulo - 90
    
    # Calcular el centro de la imagen
    (h_img, w_img) = img_crop.shape[:2]
    centro = (w_img // 2, h_img // 2)
    
    # Matriz de rotación para rotar la imagen a 0 grados (enderezar las letras)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    
    # Rotar la imagen para enderezar las letras a 0 grados
    img_rotada = cv2.warpAffine(img_crop, M, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Mostrar la imagen rotada
    cv2.imshow('Matricula Enderezada a 0 Grados', img_rotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img_rotada

def resize_image(img, max_width=760, max_height=232):
    """
    Redimensiona la imagen para que se ajuste a las dimensiones máximas especificadas.
    """
    height, width = img.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(img, (new_width, new_height)), scaling_factor

def main():
    # Configuración
    model_path = 'yolov5/runs/train/exp5/weights/best.pt'
    img_path = '12.jpeg'

    # Proceso
    model = load_model(model_path)
    img = load_image(img_path)
    detections = detect_objects(model, img)
    best_detection = get_best_detection(detections)

    if best_detection is not None:
        # Obtener coordenadas de la mejor detección
        coords = list(map(int, best_detection[:4]))


        # Recortar la imagen usando las coordenadas
        img_crop = recortar_imagen_con_coordenadas(img, coords)

        img_enderezada = procesar_y_enderezar_letras(img_crop)

        # Redimensionar la imagen enderezada
        imagen_aumentada, scaling_factor = resize_image(img_enderezada)

        directorio = 'imatges_matriculas'
        img_output_path = os.path.join(directorio, img_path)
        
        # Verificar si el directorio existe, y si no, crearlo
        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Guardar la imagen en el directorio especificado
        cv2.imwrite(img_output_path, imagen_aumentada)
        print(f"Imagen guardada en: {img_output_path}")
        
    else:
        print("No se detectó ninguna matrícula en la imagen.")

if __name__ == "__main__":
    main()