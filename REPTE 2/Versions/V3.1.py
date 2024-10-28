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

def dibujar_linea(imagen, coordenada_y_linea):
    """Dibuja una línea horizontal en el centro de la imagen que se utiliza como referencia para el conteo."""
    cv.line(imagen, (0, coordenada_y_linea), (imagen.shape[1], coordenada_y_linea), (0, 255, 0), thickness=2)

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
    """Procesa el video para detectar coches y contar vehículos que cruzan líneas en direcciones específicas."""
    contador_arriba = 0
    contador_abajo = 0
    ruta_video = "output5_tallat.mp4"  # Ruta del video a procesar
    ruta_video_tracking = "tracking_" + os.path.basename(ruta_video)  # Nombre para el video de seguimiento
    
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
                                  today=None)

    # Se dibujan dos líneas horizontales en el frame para el conteo
    linea_central_y = int(captura_video.get(cv.CAP_PROP_FRAME_HEIGHT) / 2.3)
    linea_superior_y = linea_central_y - 162  # Ajusta la distancia entre líneas según sea necesario

    # Diccionario que guarda la última posición registrada de cada track_id
    ultima_posicion = {}
    # Almacena la última dirección de cada vehículo
    ultima_direccion = {}

    # Crear el objeto VideoWriter para guardar el video procesado
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(ruta_video_tracking, fourcc, 30, (ancho_deseado, alto_deseado))

    while True:
        (exito, frame) = captura_video.read()
        if not exito:
            break

        # Se redimensiona el frame a las dimensiones especificadas
        frame = cv.resize(frame, (ancho_deseado, alto_deseado))

        # Se realiza la predicción usando el modelo YOLO para detectar objetos (solo coches - clase 2)
        resultados = modelo.predict(frame, stream=False, classes=[2])  # Detectar solo coches (clase 2)
        clases_detectadas = resultados[0].names

        # Se dibujan las líneas en el frame
        dibujar_linea(frame, linea_central_y)
        dibujar_linea(frame, linea_superior_y)

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

            # Se verifica si el rastro ha cruzado las líneas
            if track_id not in ultima_posicion:
                ultima_posicion[track_id] = coordenada_y_caja
                ultima_direccion[track_id] = None  # Inicializar dirección como None
            else:
                posicion_anterior_y = ultima_posicion[track_id]

                # Verificar si el coche está en movimiento
                desplazamiento = abs(coordenada_y_caja - posicion_anterior_y)
                if desplazamiento > 3:
                    # Verificar si el coche ha tocado alguna de las líneas
                    ha_tocado_linea = (posicion_anterior_y > linea_central_y and coordenada_y_caja <= linea_central_y) or \
                                      (posicion_anterior_y < linea_central_y and coordenada_y_caja >= linea_central_y) or \
                                      (posicion_anterior_y > linea_superior_y and coordenada_y_caja <= linea_superior_y) or \
                                      (posicion_anterior_y < linea_superior_y and coordenada_y_caja >= linea_superior_y)
    
                    if ha_tocado_linea:  # Solo si ha tocado alguna línea
                        # Se detecta si el coche ha cruzado de abajo hacia arriba
                        if posicion_anterior_y > linea_central_y and coordenada_y_caja <= linea_central_y:
                            direccion = "arriba"
                            if ultima_direccion[track_id] != "arriba":  # Verifica si la dirección ha cambiado
                                contador_arriba += 1
                                print("AAAAAAAAAAAA", track_id)
                                ultima_direccion[track_id] = direccion
    
                        # Se detecta si el coche ha cruzado de arriba hacia abajo
                        elif posicion_anterior_y < linea_central_y and coordenada_y_caja >= linea_central_y:
                            direccion = "abajo" 
                            if ultima_direccion[track_id] != "abajo":  # Verifica si la dirección ha cambiado
                                contador_abajo += 1
                                print("BBBBBBBBBBBB", track_id)
                                ultima_direccion[track_id] = direccion
    
                        # Se detecta si el coche ha cruzado la línea superior
                        if posicion_anterior_y > linea_superior_y and coordenada_y_caja <= linea_superior_y:
                            direccion = "arriba"
                            if ultima_direccion[track_id] != "arriba":  # Verifica si la dirección ha cambiado
                                contador_arriba += 1
                                print("CCCCCCCCCCCCC", track_id)
                                ultima_direccion[track_id] = direccion
    
                        elif posicion_anterior_y < linea_superior_y and coordenada_y_caja >= linea_superior_y:
                            direccion = "abajo"
                            if ultima_direccion[track_id] != "abajo":  # Verifica si la dirección ha cambiado
                                contador_abajo += 1
                                print("DDDDDDDDDDDDDDDD", track_id)
                                ultima_direccion[track_id] = direccion
    
                            # Actualiza la dirección y la última posición del rastro
                        ultima_posicion[track_id] = coordenada_y_caja

            add_text(frame, bbox, track_id, contador_arriba, contador_abajo)

        # Mostrar el frame procesado
        cv.imshow('Video Tracking', frame)

        # Guardar el frame en el video
        writer.write(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    captura_video.release()
    writer.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    procesar_video()
