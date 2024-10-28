import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

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
    """Procesa el video para detectar coches y contar vehículos que suben o bajan sin usar línea de referencia."""
    contador_arriba = 0
    contador_abajo = 0
    ruta_video = "output5_tallat8.mp4"  # Ruta del video a procesar
    ruta_video_tracking = "trackingV8_" + os.path.basename(ruta_video)  # Nombre para el video de seguimiento

    # Se lee el video
    captura_video = cv.VideoCapture(ruta_video)

    # Se establecen las dimensiones deseadas para el frame del video
    ancho_deseado = 360
    alto_deseado = 640

    # Se carga el modelo YOLO
    modelo = YOLO('yolov8n.pt')

    rastreador_objetos = DeepSort(max_iou_distance=0.9,
                                  max_age=5,
                                  n_init=2,
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
                                  today=None)

    # Diccionario que guarda la última posición registrada de cada track_id
    ultima_posicion = {}
    # Diccionario que guarda la cuenta de frames de movimiento continuo para cada track_id
    frames_movimiento_continuo = {}

    # Almacena la última dirección de cada vehículo
    ultima_direccion = {}

    # Crear el objeto VideoWriter para guardar el video procesado
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(ruta_video_tracking, fourcc, 30, (ancho_deseado, alto_deseado))

    # Contador de frames
    frame_count = 0

    margen_movimiento = 70  # Umbral para considerar si un coche se mueve lo suficiente

    while True:
        (exito, frame) = captura_video.read()
        if not exito:
            break

        # Incrementar el contador de frames
        frame_count += 1

        # Solo procesar un frame de cada 10
        if frame_count % 10 != 0:
            continue

        # Se redimensiona el frame a las dimensiones especificadas
        frame = cv.resize(frame, (ancho_deseado, alto_deseado))

        # Se realiza la predicción usando el modelo YOLO para detectar objetos (solo coches - clase 2)
        resultados = modelo.predict(frame, stream=False, classes=[2])  # Detectar solo coches (clase 2)
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
            coordenada_y_caja = bbox[3]  # Coordenada y inferior de la caja delimitadora
            
            # Verificar si el coche ha sido detectado antes
            if track_id not in ultima_posicion:
                ultima_posicion[track_id] = coordenada_y_caja
                frames_movimiento_continuo[track_id] = 0
                ultima_direccion[track_id] = None
            else:
                posicion_anterior_y = ultima_posicion[track_id]

                # Verificar si el coche está en movimiento
                desplazamiento = abs(coordenada_y_caja - posicion_anterior_y)
                print(track_id, desplazamiento)
                if desplazamiento > margen_movimiento:
                    frames_movimiento_continuo[track_id] += 1
                    if frames_movimiento_continuo[track_id] >= 3:
                        # Determinar si el coche se mueve hacia arriba o hacia abajo
                        if posicion_anterior_y > coordenada_y_caja:
                            direccion = "arriba"
                            if ultima_direccion[track_id] != "arriba":
                                contador_arriba += 1
                                ultima_direccion[track_id] = direccion
                                print("Sube", track_id)
                        elif posicion_anterior_y < coordenada_y_caja:
                            direccion = "abajo"
                            if ultima_direccion[track_id] != "abajo":
                                contador_abajo += 1
                                ultima_direccion[track_id] = direccion
                                print("Baja", track_id)

                        ultima_posicion[track_id] = coordenada_y_caja
                else:
                    frames_movimiento_continuo[track_id] = 0

            # Añade el texto con la información del coche
            add_text(frame, bbox, track_id, contador_arriba, contador_abajo)

        # Mostrar el frame procesado
        cv.imshow('Video Tracking', frame)

        # Escribir el frame en el video de salida
        writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    captura_video.release()
    writer.release()
    cv.destroyAllWindows()

    # Resultados finales
    print('Count Up:', contador_arriba)
    print('Count Down:', contador_abajo)

# Llamar a la función para procesar el video
procesar_video()
