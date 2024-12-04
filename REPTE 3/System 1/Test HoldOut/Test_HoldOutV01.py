import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from Models.AEmodels import AutoEncoderCNN
import os
import glob
import cv2

# Establece la variable de entorno antes de importar cualquier otra librería
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
inputmodule_paramsEnc = {'num_input_channels': 3}

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_holdout_labels(csv_path):
    """
    Carga los IDs y etiquetas del conjunto HoldOut desde un archivo CSV.
    
    Args:
        csv_path: Ruta al archivo CSV con columnas 'CODI' y 'DENSITAT'.
        
    Returns:
        holdout_ids: Lista de IDs de pacientes.
        holdout_labels: Lista de etiquetas numéricas correspondientes.
    """
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Convertir las etiquetas de 'DENSITAT' a valores numéricos
    label_mapping = {
        "ALTA": 1,
        "BAIXA": 1,
        "NEGATIVA": 0
    }
    df['DENSITAT'] = df['DENSITAT'].map(label_mapping)
    
    # Extraer IDs y etiquetas
    holdout_ids = df['CODI'].tolist()
    holdout_labels = df['DENSITAT'].tolist()

    return holdout_ids, holdout_labels

# Configuración del Autoencoder
def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    net_paramsEnc['drop_rate'] = 0.3
    net_paramsDec['drop_rate'] = 0.3
    
    if Config == '1':
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

# Cargar el modelo
def load_autoencoder_model():
    Config = '1'
    inputmodule_paramsEnc = {'num_input_channels': 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    model.load_state_dict(torch.load('best_autoencoder_model.pth', map_location=device))
    model.eval()
    model.to(device)
    return model

# Función para calcular el error F_red entre la imagen original y la reconstruida
def calculate_error_F_red(original_image, reconstructed_image):
    hsv_original = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_reconstructed = cv2.cvtColor((reconstructed_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    H_ori = hsv_original[:, :, 0]
    H_rec = hsv_reconstructed[:, :, 0]
    
    lower_threshold_1, upper_threshold_1 = 0, 20
    lower_threshold_2, upper_threshold_2 = 160, 179
    
    mask_ori = ((H_ori >= lower_threshold_1) & (H_ori <= upper_threshold_1)) | \
               ((H_ori >= lower_threshold_2) & (H_ori <= upper_threshold_2))
    mask_rec = ((H_rec >= lower_threshold_1) & (H_rec <= upper_threshold_1)) | \
               ((H_rec >= lower_threshold_2) & (H_rec <= upper_threshold_2))
    
    F_red = np.sum(mask_ori) / max(np.sum(mask_rec), 1)
    
    return F_red

# Función para procesar imágenes de un paciente y calcular F_red
def process_patient_images(patient_id, data_folder, model):
    patient_folders = glob.glob(os.path.join(data_folder, f"{patient_id}_*"))
    F_red_list = []
    
    for patient_folder in patient_folders:
        image_files = [f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(patient_folder, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image_rgb, (256, 256))
                normalized_image = resized_image / 255.0
                input_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    reconstructed_image = model(input_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                F_red = calculate_error_F_red(normalized_image, reconstructed_image)
                F_red_list.append(F_red)
    return F_red_list

# Evaluar HoldOut
def evaluate_holdout_set(holdout_ids, holdout_labels, data_folder, model, patch_threshold, patient_threshold):
    true_labels = []
    predicted_labels = []

    # Usar una copia de los IDs y etiquetas para modificarlos en el bucle
    filtered_holdout_ids = holdout_ids[:]
    filtered_holdout_labels = holdout_labels[:]

    for patient_id, true_label in zip(filtered_holdout_ids, filtered_holdout_labels):
        F_red_list = process_patient_images(patient_id, data_folder, model)
        total_patches = len(F_red_list)

        # Si no hay parches para el paciente, eliminarlo y continuar con el siguiente
        if total_patches == 0:
            print(f"No images found for patient {patient_id}.")
            continue

        # Calcular el número de parches positivos y la predicción del paciente
        positive_patches = sum(1 for f_red in F_red_list if f_red > patch_threshold)
        prediction_ratio = positive_patches / total_patches
        predicted_label = 1 if prediction_ratio >= patient_threshold else 0

        # Agregar resultados a las listas
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    if len(true_labels) == 0 or len(predicted_labels) == 0:  # Validar que hay datos para procesar
        print("Error: No valid data available for evaluation.")
        return None

    # Cálculo de métricas por clase
    unique_classes = np.unique(true_labels)
    recall_per_class = recall_score(true_labels, predicted_labels, labels=unique_classes, average=None, zero_division=0)
    precision_per_class = precision_score(true_labels, predicted_labels, labels=unique_classes, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predicted_labels, labels=unique_classes, average=None, zero_division=0)

    metrics = {cls: {"recall": float(recall), "precision": float(precision), "f1": float(f1)}
               for cls, recall, precision, f1 in zip(unique_classes, recall_per_class, precision_per_class, f1_per_class)}

    total_correct = np.sum(true_labels == predicted_labels)
    total_samples = len(true_labels)
    accuracy_global = total_correct / total_samples

    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for cls, acc in zip(unique_classes, accuracy_per_class):
        metrics[cls]["accuracy"] = float(acc)

    # Imprimir métricas por clase
    print("\nMetrics Summary:")
    for cls, cls_metrics in metrics.items():
        print(f"\nClass {cls}:")
        for metric_name, metric_value in cls_metrics.items():
            print(f"  {metric_name.capitalize()}: {metric_value:.4f}")

    # Gráfica Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title("Confusion Matrix - HoldOut Set")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/HoldOut_Confusion_MatrixV01.png")
    plt.close()

    # Calcular métricas globales
    recall_global = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
    precision_global = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
    f1_global = f1_score(true_labels, predicted_labels, average='binary')
    accuracy_global = accuracy_score(true_labels, predicted_labels)

    global_metrics = {
        "recall": recall_global,
        "precision": precision_global,
        "f1": f1_global,
        "accuracy": accuracy_global
    }

    print("\nGlobal Metrics:")
    for metric, value in global_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    # Crear un gráfico de barras para las métricas globales
    metrics_names = list(global_metrics.keys())
    avg_values = list(global_metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_names, avg_values, color='skyblue', alpha=0.8)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Global Metrics Across HoldOut Set")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/HoldOut_Global_MetricsV01.png")
    plt.close()

    print("Metrics plot saved to /export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/HoldOut_Global_MetricsV01.png")

    return metrics



# Parámetros y ejecución
patch_threshold = 221.6
patient_threshold = 0.46725684795553785

csv_path = "/export/fhome/maed01/HelicoDataSet/PatientDiagnosis.csv"
holdout_ids, holdout_labels = load_holdout_labels(csv_path)

data_folder = "/export/fhome/maed01/HelicoDataSet/HoldOut"
model = load_autoencoder_model()

metrics = evaluate_holdout_set(holdout_ids, holdout_labels, data_folder, model, patch_threshold, patient_threshold)
