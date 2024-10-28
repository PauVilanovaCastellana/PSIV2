"""
VERSIO AMB LINIA
"""

import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    
def add_text(frame, bbox, track_id, contador_arriba, contador_abajo, offset = 10):
    """
    Añade el texto en cada Frame (ID cotches y conteo Up/Down)
    """
    # Calcular el centro X para el ID del vehículo
    centro_x = int((bbox[0] + bbox[2]) // 2)  - offset*6 # No se usa offset, se centra directamente
    pos_y = int(bbox[1]) - offset  # Posición Y ajustada

    # Dibujar el ID del vehículo en el frame
    cv.putText(frame, "ID: " + str(track_id), (centro_x, pos_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Obtener las dimensiones del frame
    altura, anchura = frame.shape[:2]

    # Calcular posiciones para los contadores
    pos_count_up_y = altura - 20  # Ajusta este valor para colocar el texto donde desees
    pos_count_down_y = altura - 70  # Ajusta este valor para colocar el texto donde desees

    # Dibujar los contadores en el frame
    cv.putText(frame, "Count Up: " + str(contador_arriba), (5, pos_count_down_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv.putText(frame, "Count Down: " + str(contador_abajo), (5, pos_count_up_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)


def dibujar_linea(imagen, coordenada_y_linea):
    """
    Dibuja una línea horizontal en el centro de la imagen que se utiliza como referencia para el conteo.
    """
    cv.line(imagen, (0, coordenada_y_linea), (imagen.shape[1], coordenada_y_linea), (0, 255, 0), thickness=2)

def dibujar_caja(data, imagen, nombre_clase):
    """
    Dibuja una caja delimitadora alrededor del objeto detectado y la etiqueta con el nombre de la clase.
    """
    x1, y1, x2, y2, confianza, id_clase = data
    punto1 = (int(x1), int(y1))
    punto2 = (int(x2), int(y2))
    cv.rectangle(imagen, punto1, punto2, (0, 0, 255), 3)
    return imagen

def obtener_detalles(resultado, imagen):
    """
    Extrae los detalles de las detecciones, incluyendo las coordenadas, confianza y clase de los objetos detectados.
    """
    clases = resultado.boxes.cls.numpy()
    confianza = resultado.boxes.conf.numpy()
    coordenadas_xywh = resultado.boxes.xywh.numpy()

    detecciones = []
    for i, item in enumerate(coordenadas_xywh):
        muestra = (item, confianza[i], clases[i])
        detecciones.append(muestra)

    return detecciones

def procesar_video():
    """
    Procesa el video para detectar objetos y contar vehículos que cruzan una línea en una dirección específica.
    """
    cache_contador = set()  # Almacena los track_id ya contados
    clases_detectadas = []
    contador_arriba = 0
    contador_abajo = 0
    ruta_video = "output7.mp4"  # Ruta del video a procesar
    
    # Se lee el video
    captura_video = cv.VideoCapture(ruta_video)
    
    # Se establecen las dimensiones deseadas para el frame del video
    ancho_deseado = 360
    alto_deseado = 640

    # Se carga el modelo YOLO
    modelo = YOLO('yolov8n.pt')

    rastreador_objetos = DeepSort(max_iou_distance=0.7,
                                  max_age=5,
                                  n_init=3,
                                  nms_max_overlap=1.0,
                                  max_cosine_distance=0.2,
                                  nn_budget=None,
                                  gating_only_position=False,
                                  override_track_class=None,
                                  embedder="mobilenet",
                                  half=True,
                                  bgr=True,
                                  embedder_gpu=True,
                                  embedder_model_name=None,
                                  embedder_wts=None,
                                  polygon=False,
                                  today=None
                                  )

    # Se dibuja una línea horizontal en el centro del frame para el conteo
    linea_central_y = int(captura_video.get(cv.CAP_PROP_FRAME_HEIGHT) / 2)
    
    # Diccionario que guarda la última posición registrada de cada track_id
    ultima_posicion = {}

    while True:
        (exito, frame) = captura_video.read()
        if not exito:
            break

        # Se redimensiona el frame a las dimensiones especificadas
        frame = cv.resize(frame, (ancho_deseado, alto_deseado))

        # Se realiza la predicción usando el modelo YOLO para detectar objetos (clases 2 y 7: vehículos)
        resultados = modelo.predict(frame, stream=False, classes=[2, 7])
        clases_detectadas = resultados[0].names
        
        # Se dibuja la línea central en el frame
        dibujar_linea(frame, linea_central_y)

        for resultado in resultados:
            for data in resultado.boxes.data.tolist():
                id_clase = data[5]
                dibujar_caja(data, frame, clases_detectadas[id_clase])

            detalles = obtener_detalles(resultado, frame)
            rastros = rastreador_objetos.update_tracks(detalles, frame=frame)

        for rastro in rastros:
            if not rastro.is_confirmed():
                continue
            
            track_id = rastro.track_id
            bbox = rastro.to_ltrb()
            coordenada_y_caja = bbox[1]  # Coordenada y superior de la caja delimitadora

            # Se verifica si el rastro ha cruzado la línea
            if track_id not in ultima_posicion:
                ultima_posicion[track_id] = coordenada_y_caja
            else:
                posicion_anterior_y = ultima_posicion[track_id]
                
                # Se detecta si el vehículo ha cruzado de abajo hacia arriba
                if posicion_anterior_y > linea_central_y and coordenada_y_caja <= linea_central_y and track_id not in cache_contador:
                    contador_arriba += 1
                    cache_contador.add(track_id)
                # Se detecta si el vehículo ha cruzado de arriba hacia abajo
                elif posicion_anterior_y < linea_central_y and coordenada_y_caja >= linea_central_y and track_id not in cache_contador:
                    contador_abajo += 1
                    cache_contador.add(track_id)

                # Se actualiza la última posición del rastro
                ultima_posicion[track_id] = coordenada_y_caja

            add_text(frame, bbox, track_id, contador_arriba, contador_abajo)
            
        # Se muestra el frame procesado
        cv.imshow('image', frame)
        cv.waitKey(1)

    print('Count Up:', contador_arriba)
    print('Count Down:', contador_abajo)


##### MAIN #####
procesar_video()
