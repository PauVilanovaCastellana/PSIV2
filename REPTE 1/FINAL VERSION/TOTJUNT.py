from Localitzacio import *
from Segmentacio import *
from Reconeixement import *

def localizar_matricula(model_path, img_path, output_path, margen=60):
    """
    Procesa una única imagen utilizando el modelo YOLO y guarda la imagen procesada en la ruta de salida.
    """
    # Cargar el modelo YOLO
    model = load_model_loc(model_path)
    
    # Cargar la imagen
    img = load_image(img_path)
    
    # Detectar objetos (matrículas) en la imagen
    detections = detect_objects(model, img)
    best_detection = get_best_detection(detections)

    if best_detection is not None:
        # Obtener coordenadas de la mejor detección
        coords = list(map(int, best_detection[:4]))

        # Recortar la imagen usando las coordenadas con margen
        img_crop = recortar_imagen_con_coordenadas(img, coords, 10)

        # Procesar y enderezar las letras
        img_enderezada = procesar_y_enderezar_letras(img_crop)

        # Redimensionar la imagen enderezada
        imagen_aumentada, scaling_factor = resize_image(img_enderezada)

        # Intentar una segunda detección en la imagen redimensionada
        detections_resized = detect_objects(model, imagen_aumentada)
        best_detection_resized = get_best_detection(detections_resized)

        if best_detection_resized is not None:
            # Obtener coordenadas de la detección redimensionada
            x_min, y_min, x_max, y_max = map(int, best_detection_resized[:4])

            # Recortar la imagen con margen usando las coordenadas redimensionadas
            coords_resized = [x_min, y_min, x_max, y_max]
            img_crop_con_margen = recortar_imagen_con_coordenadas(imagen_aumentada, coords_resized, margen=margen)

            # Guardar la imagen recortada con margen
            cv2.imwrite(output_path, img_crop_con_margen)
            print(f"Imagen con margen guardada en: {output_path}")
        else:
            # Guardar la imagen redimensionada si no se detecta ninguna matrícula
            print(f"No se detectó matrícula tras redimensionar: {img_path}")
            cv2.imwrite(output_path, imagen_aumentada)
            print(f"Imagen redimensionada guardada en: {output_path}")
    else:
        print(f"No se detectó ninguna matrícula en: {img_path}")

def do_reconeixement(input_path_reconeixement, path_reconeixement='Models_Reconeixement/'):
    """
    Realiza el reconocimiento de números y letras en las imágenes procesadas.
    """
    # Cargar los modelos guardados
    number_model = load_model(path_reconeixement + 'clasificador_numeros_33x47.keras')  # Modelo para los números
    letter_feature_extractor = load_model(path_reconeixement + 'letter_feature_extractor_model.keras')  # Red de extracción de características para letras
    letter_svm = joblib.load(path_reconeixement + 'letter_svm_classifier_model.keras')  # Clasificador SVM para letras

    input_path_reconeixement = os.path.join(output_path_segmentacio, img_path.split('.')[0])

    # Cargar y preprocesar las imágenes para predecir los números
    number_images = []
    for img_name in sorted(os.listdir(input_path_reconeixement))[:4]:  # Las primeras 4 imágenes
        img_path_full = os.path.join(input_path_reconeixement, img_name)
        number_images.append(preprocess_image(img_path_full, 33, 47))  # Preprocesar para el modelo de números

    predicted_numbers = predict_numbers(number_model, number_images)
    predicted_number_str = ''.join(map(str, predicted_numbers))  # Convertir las predicciones a string

    # Cargar y preprocesar las últimas 3 imágenes para predecir las letras
    letter_images = []
    for img_name in sorted(os.listdir(input_path_reconeixement))[-3:]:  # Las últimas 3 imágenes
        img_path_full = os.path.join(input_path_reconeixement, img_name)
        img = cv2.imread(img_path_full, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 30))  # Preprocesar para el modelo de letras
        letter_images.append(img)

    predicted_letter_features = letter_feature_extractor.predict(np.array(letter_images).reshape(-1, 30, 30, 1))
    predicted_letters = letter_svm.predict(predicted_letter_features)
    predicted_letter_str = ''.join(predicted_letters)

    # Predicción final para la matrícula completa
    prediccion_final = predicted_number_str + predicted_letter_str

    # Imprimir la predicción final
    print('Predicción final:', prediccion_final)

if __name__ == "__main__":
    ###### 1.LOCALIZACION ######
    print("- - - - 1. START LOCALIZACION - - - - ")
    # Configuración
    model_path = 'yolov5/runs/train/exp5/weights/best.pt'
    img_path = '1.jpg'  # Cambia esto a la ruta de tu imagen
    output_path_localitzacio = 'imatges_matriculas/'+ img_path  # Ruta donde se guardará la imagen procesada

    localizar_matricula(model_path, img_path, output_path_localitzacio)
    
    ###### 2.SEGMENTACION ######
    print("\n")
    print("- - - - 2. START SEGMENTACION - - - - ")
    output_path_segmentacio = 'matricula_segmentada'
    procesar_imagen(output_path_localitzacio, output_path_segmentacio)
    
    ###### 3.RECONOCIMIENTO ######
    print("\n")
    print("- - - - 3. START RECONOCIMIENTO - - - - ")
    # Cargar los modelos guardados
    input_path_reconeixement = output_path_segmentacio + "/" + img_path.split('.')[0]
    do_reconeixement(input_path_reconeixement)
