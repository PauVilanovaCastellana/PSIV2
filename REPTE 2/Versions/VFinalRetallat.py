# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:46:31 2024

@author: eduar
"""

import time  # Importa el módulo time

import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def add_text(frame, bbox, track_id, contador_arriba, contador_abajo, offset=10):
    """Añade el texto en cada Frame (ID coches y conteo Up/Down)"""
    centro_x = int((bbox[0] + bbox[2]) // 2) - offset * 6
    pos_y = int(bbox[1]) - offset
    cv.putText(frame, "ID: " + str(track_id), (centro_x, pos_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    altura, anchura = frame.shape[:2]
    pos_count_up_y = altura - 20
    pos_count_down_y = altura - 70
    cv.putText(frame, "Count Up: " + str(contador_arriba), (5, pos_count_down_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    cv.putText(frame, "Count Down: " + str(contador_abajo), (5, pos_count_up_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

def dibujar_caja(data, imagen, nombre_clase):
    """Dibuja una caja delimitadora alrededor del objeto detectado y la etiqueta con el nombre de la clase."""
    x1, y1, x2, y2, confianza, id_clase = data
    punto1 = (int(x1), int(y1))
    punto2 = (int(x2), int(y2))
    cv.rectangle(imagen, punto1, punto2, (0, 0, 255), 3)
    return imagen

def obtener_detalles(resultado, imagen):
    """Extrae los detalles de las detecciones, incluyendo las coordenadas, confianza y clase de los objetos detectados."""
    clases = resultado.boxes.cls.numpy()
    confianza = resultado.boxes.conf.numpy()
    coordenadas_xywh = resultado.boxes.xywh.numpy()

    detecciones = []
    for i, item in enumerate(coordenadas_xywh):
        muestra = (item, confianza[i], clases[i])
        detecciones.append(muestra)

    return detecciones

def procesar_video():
    tiempo_inicio = time.time()
    contador_arriba = 0
    contador_abajo = 0
    ruta_video = "output5.mp4"
    ruta_video_tracking = "trackingVFinal2Retallat_" + os.path.basename(ruta_video)

    captura_video = cv.VideoCapture(ruta_video)
    ancho_deseado = 360
    alto_deseado = 640

    modelo = YOLO('yolov8n.pt')
    rastreador_objetos = DeepSort(max_iou_distance=0.9, max_age=5, n_init=2, nms_max_overlap=1.0,
                                  max_cosine_distance=0.2, nn_budget=None, gating_only_position=False,
                                  override_track_class=None, embedder="mobilenet", half=True, bgr=True,
                                  embedder_gpu=True, embedder_model_name=None, embedder_wts=None,
                                  polygon=False, today=None)

    linea_conteo_y = 515
    margen_movimiento = 48

    ultima_posicion = {}
    frames_movimiento_continuo = {}
    ultima_direccion = {}
    objetos_tocaron_linea = {}

    # Definimos los puntos para la línea diagonal
    punto_inicio_diagonal = (280, 210)  # Punto en el borde superior derecho
    punto_fin_diagonal = (ancho_deseado, alto_deseado) 
    limite_frames_linea_diagonal = 10

    # Diccionario para almacenar los frames en la línea diagonal
    frames_en_linea_diagonal = {}

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(ruta_video_tracking, fourcc, 30, (ancho_deseado - 40, alto_deseado -40))

    frame_count = 0
    margen = 25

    while True:
        (exito, frame) = captura_video.read()
        if not exito:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        frame = cv.resize(frame, (ancho_deseado, alto_deseado))
        frame = frame[40:,:]
        alto_deseado_recortado = frame.shape[0]
        ancho_deseado_recortado = frame.shape[1]

        # Línea de conteo horizontal
        cv.line(frame, (0, linea_conteo_y), (ancho_deseado_recortado, linea_conteo_y), (0, 255, 255), 2)
        cv.line(frame, (0, linea_conteo_y + margen), (ancho_deseado_recortado, linea_conteo_y + margen), (128, 128, 128), 2)
        cv.line(frame, (0, linea_conteo_y - margen), (ancho_deseado_recortado, linea_conteo_y - margen), (128, 128, 128), 2)

        # Dibuja la línea diagonal
        cv.line(frame, punto_inicio_diagonal, punto_fin_diagonal, (255, 0, 0), 2)

        resultados = modelo.predict(frame, stream=False, classes=[2], verbose=False)
        clases_detectadas = resultados[0].names

        for resultado in resultados:
            detalles = obtener_detalles(resultado, frame)
            rastros = rastreador_objetos.update_tracks(detalles, frame=frame)

            for data in resultado.boxes.data.tolist():
                id_clase = data[5]
                dibujar_caja(data, frame, clases_detectadas[id_clase])

        for rastro in rastros:
            if not rastro.is_confirmed():
                continue

            track_id = rastro.track_id
            bbox = rastro.to_ltrb()
            coordenada_y_caja = bbox[3]
            coordenada_x_caja = bbox[2]  # Coordenada X de la esquina derecha de la caja

            # Calcular la distancia del vehículo a la línea diagonal
            dist = np.abs((punto_fin_diagonal[1] - punto_inicio_diagonal[1]) * coordenada_x_caja -
                          (punto_fin_diagonal[0] - punto_inicio_diagonal[0]) * coordenada_y_caja +
                          punto_fin_diagonal[0] * punto_inicio_diagonal[1] -
                          punto_fin_diagonal[1] * punto_inicio_diagonal[0]) / \
                   np.sqrt((punto_fin_diagonal[1] - punto_inicio_diagonal[1])**2 +
                           (punto_fin_diagonal[0] - punto_inicio_diagonal[0])**2)

            # Si la distancia es pequeña, contar los frames que ha estado en esta zona
            if dist < 10:
                frames_en_linea_diagonal[track_id] = frames_en_linea_diagonal.get(track_id, 0) + 1
                if frames_en_linea_diagonal[track_id] >= limite_frames_linea_diagonal:
                    continue  # Si sobrepasa el límite, no contar este vehículo
            else:
                frames_en_linea_diagonal[track_id] = 0

            # Verificar movimiento y contar solo si no está en la línea diagonal por el tiempo límite
            if track_id not in ultima_posicion:
                ultima_posicion[track_id] = coordenada_y_caja
                frames_movimiento_continuo[track_id] = 0
                ultima_direccion[track_id] = None
                objetos_tocaron_linea[track_id] = False
            else:
                posicion_anterior_y = ultima_posicion[track_id]
                desplazamiento = abs(coordenada_y_caja - posicion_anterior_y)

                if not objetos_tocaron_linea[track_id] and \
                   (linea_conteo_y - margen <= coordenada_y_caja <= linea_conteo_y + margen):
                    objetos_tocaron_linea[track_id] = True

                if desplazamiento > margen_movimiento:
                    frames_movimiento_continuo[track_id] += 1
                    if frames_movimiento_continuo[track_id] >= 4:
                        if objetos_tocaron_linea[track_id]:
                            if posicion_anterior_y > linea_conteo_y >= coordenada_y_caja:
                                direccion = "arriba"
                                if ultima_direccion[track_id] != "arriba":
                                    contador_arriba += 1
                                    ultima_direccion[track_id] = direccion

                            elif posicion_anterior_y < linea_conteo_y <= coordenada_y_caja:
                                direccion = "abajo"
                                if ultima_direccion[track_id] != "abajo":
                                    contador_abajo += 1
                                    ultima_direccion[track_id] = direccion

                        ultima_posicion[track_id] = coordenada_y_caja
                else:
                    frames_movimiento_continuo[track_id] = 0

            add_text(frame, bbox, track_id, contador_arriba, contador_abajo)

        writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    captura_video.release()
    writer.release()
    
    tiempo_final = time.time()
    duracion = tiempo_final - tiempo_inicio
    print(f'Tiempo total de ejecución: {duracion:.2f} segundos')

    print('Count Up:', contador_arriba)
    print('Count Down:', contador_abajo)



# Llamar a la función para procesar el video
procesar_video()
