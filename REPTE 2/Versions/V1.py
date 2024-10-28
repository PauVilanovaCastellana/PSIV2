import cv2
import torch

# Carregar el model YOLOv5 preentrenat
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pots provar altres models de YOLOv5

# Iniciar Sort tracker
tracker = Sort()

# Carregar el vídeo
video = cv2.VideoCapture('output7.mp4')

# Comptadors
vehicles_direccio_amunt = 0
vehicles_direccio_avall = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detecció amb YOLO
    deteccions = model(frame)  # Retorna deteccions en forma de DataFrame
    
    # Filtrar solament cotxes
    deteccions = deteccions.pandas().xyxy[0]
    cotxes = deteccions[deteccions['name'] == 'car']

    # Convertir deteccions a format compatible amb Sort (x1, y1, x2, y2, score)
    bbox = cotxes[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values

    # Tracking dels cotxes
    tracks = tracker.update(bbox)

    # Aquí pots implementar la lògica per dividir en carrils i comptar vehicles
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        # Condicions per a comptar vehicles que van cap amunt o avall
        # (Depenent de la posició dels vehicles al frame i si creuen la línia)
        
        # Mostra les deteccions i el tracking
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    cv2.imshow('Detecció i Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Alliberar recursos
video.release()
cv2.destroyAllWindows()
