import os
import glob
import pandas as pd
import cv2
import torch
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from Models.weights_init import weights_init_xavier, weights_init_kaiming
from sklearn.model_selection import GroupKFold



# Establece la variable de entorno antes de importar cualquier otra librería
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
inputmodule_paramsEnc = {'num_input_channels': 3}

# Cargar los IDs y etiquetas de los primeros k pacientes únicos
def load_patient_labels(file_path, k=154):
    df = pd.read_excel(file_path)
    print("Columnas disponibles en el archivo Excel:", df.columns)
    
    if 'Pat_ID' not in df.columns or 'Presence' not in df.columns:
        raise KeyError("El archivo Excel debe contener las columnas 'Pat_ID' y 'Presence'.")
    
    patient_ids = df['Pat_ID'].tolist()
    true_labels = df['Presence'].tolist()
    
    # Seleccionar los primeros k pacientes únicos
    unique_patients = list(set(patient_ids))[:k]
    k_patient_ids = []
    k_labels = []
    for patient1 in unique_patients:
        for patient2, label in zip(patient_ids, true_labels):
            if patient1 == patient2:
                k_patient_ids.append(patient2)
                k_labels.append(label)
    
    return k_patient_ids, k_labels

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

# Cargar el modelo (CPU)
def load_autoencoder_model():
    Config = '1'
    inputmodule_paramsEnc = {'num_input_channels': 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    model.load_state_dict(torch.load('autoencoder_model_V05.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Función para calcular el error F_red entre la imagen original y la reconstruida
def calculate_error_F_red(original_image, reconstructed_image):
     # Convertir las imágenes de [0, 1] a [0, 255] y luego al espacio HSV
    hsv_original = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_reconstructed = cv2.cvtColor((reconstructed_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Extraer el canal H (Hue)
    H_ori = hsv_original[:, :, 0]
    H_rec = hsv_reconstructed[:, :, 0]
    
    # Rango para tonos rojos: -20 a 20 (equivalente a 0-20 y 160-179 en OpenCV)
    lower_threshold_1, upper_threshold_1 = 0, 20  # 0 a 20
    lower_threshold_2, upper_threshold_2 = 160, 179  # 160 a 179
    
    # Máscaras para el rango -20 a 20 en el canal H
    mask_ori = ((H_ori >= lower_threshold_1) & (H_ori <= upper_threshold_1)) | \
               ((H_ori >= lower_threshold_2) & (H_ori <= upper_threshold_2))
    
    mask_rec = ((H_rec >= lower_threshold_1) & (H_rec <= upper_threshold_1)) | \
               ((H_rec >= lower_threshold_2) & (H_rec <= upper_threshold_2))
    
    # Cálculo del error F_red (evitando división por cero)
    F_red = np.sum(mask_ori) / max(np.sum(mask_rec), 1)
    
    return F_red

# Función para procesar imágenes de un paciente y calcular F_red
def process_patient_images(patient_id, data_folder, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Detecta GPU o CPU
    model.to(device)  # Asegúrate de que el modelo esté en el dispositivo correcto
    patient_folders = glob.glob(os.path.join(data_folder, f"{patient_id}_*"))
    F_red_list = []
    
    for patient_folder in patient_folders:
        image_files = [f for f in os.listdir(patient_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(patient_folder, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Preprocesamiento de la imagen
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image_rgb, (256, 256))
                normalized_image = resized_image / 255.0
                input_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Reconstrucción con el modelo
                with torch.no_grad():
                    reconstructed_image = model(input_tensor).squeeze(0).permute(1, 2, 0)  # Mantener el tensor en GPU

                # Para calcular F_red, mueve los tensores necesarios a la CPU para usarlos con OpenCV
                F_red = calculate_error_F_red(
                    normalized_image, 
                    reconstructed_image.cpu().numpy()  # Convertir solo en este punto para cálculos con NumPy/OpenCV
                )
                F_red_list.append(F_red)
    
    return F_red_list

# Evaluar el sistema y calcular el umbral óptimo mediante ROC
def evaluate_patch_classification(k_patient_ids, k_labels, data_folder, model):
    roc_save_path = "/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/PatchClassification_k_154_Validation04.png"
    boxplot_save_path = "/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/F_Red_Distribution_Boxplot_Validation04.png"
    
    all_F_red, all_labels = [], []
    
    # Procesar imágenes y recopilar errores y etiquetas
    for patient_id, label in zip(k_patient_ids, k_labels):
        F_red_list = process_patient_images(patient_id, data_folder, model)
        all_F_red.extend(F_red_list)
        all_labels.extend([label] * len(F_red_list))

    # Depuración: verificar variabilidad en all_F_red y all_labels
    print("Resumen de F_red:", np.unique(all_F_red, return_counts=True))
    print("Resumen de labels:", np.unique(all_labels, return_counts=True))

    # Verificar si hay suficiente variabilidad en las etiquetas
    if len(np.unique(all_labels)) < 2:
        print("Error: Necesitamos ejemplos de ambas clases (positiva y negativa) para calcular la curva ROC.")
        return None, None

    # Crear la distribución de F_red como boxplot
    grouped_F_red = [
        [F_red for F_red, label in zip(all_F_red, all_labels) if label == -1],  # No Helico (-1)
        [F_red for F_red, label in zip(all_F_red, all_labels) if label == 1],   # Helico (1)
        [F_red for F_red, label in zip(all_F_red, all_labels) if label == 0]    # Not clear (0)
    ]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped_F_red, labels=["No Helico (-1)", "Helico (1)", "Not clear (0)"], patch_artist=True)
    plt.xlabel('Label')
    plt.ylabel('F_red Error')
    plt.title('Distribution of F_red Errors by Label')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(boxplot_save_path, bbox_inches='tight')
    plt.close()
    print(f"Boxplot guardado en {boxplot_save_path}")

    # Calcular y graficar la curva ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_F_red, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal Th = {optimal_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patch Classification ROC')
    plt.legend(loc="lower right")
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()
    print(f"Curva ROC guardada en {roc_save_path}")

    return optimal_threshold, roc_auc

# Evaluar la clasificación a nivel de paciente (Patient Diagnosis)
def evaluate_patient_diagnosis(k_patient_ids, k_labels, data_folder, model, patch_threshold):
    save_path= "/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/PatientDiagnosis_k_154_Validation04.png"
    """
    Realiza Patient Diagnosis utilizando Adaptive Thresholding en el porcentaje de patches positivos.
    """
    patient_labels = []
    patient_predictions = []
    for patient_id, label in zip(k_patient_ids, k_labels):
        F_red_list = process_patient_images(patient_id, data_folder, model)
        positive_patches = sum(1 for f_red in F_red_list if f_red > patch_threshold)
        total_patches = len(F_red_list)
        prediction = positive_patches / total_patches
        patient_predictions.append(prediction)
        patient_labels.append(label)
    
    # Calcular curva ROC a nivel de paciente
    fpr, tpr, thresholds = roc_curve(patient_labels, patient_predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    patient_threshold = thresholds[optimal_idx]
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal Patient Th = {patient_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient Diagnosis ROC')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return patient_threshold, roc_auc, patient_predictions

# Validación cruzada del sistema
def cross_validate_system(patient_ids, labels, data_folder, model, n_splits=5, save_plot_path="/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Metrics_154_Validation04.png"):
    gkf = GroupKFold(n_splits=n_splits)
    auc_scores_patch, auc_scores_patient = [], []
    metrics_results = []

    patient_ids = np.array(patient_ids)
    labels = np.array(labels)

    for train_idx, test_idx in gkf.split(patient_ids, labels, groups=patient_ids):
        train_ids, train_labels = patient_ids[train_idx], labels[train_idx]
        test_ids, test_labels = patient_ids[test_idx], labels[test_idx]

        # Patch Classification
        patch_threshold, patch_auc = evaluate_patch_classification(train_ids, train_labels, data_folder, model)
        auc_scores_patch.append(patch_auc)

        # Patient Diagnosis
        patient_threshold, patient_auc, test_preds = evaluate_patient_diagnosis(test_ids, test_labels, data_folder, model, patch_threshold)
        auc_scores_patient.append(patient_auc)

        # Convert predictions to binary
        test_preds_binary = [1 if pred >= patient_threshold else 0 for pred in test_preds]

        # Calculate metrics with proper handling
        recall = recall_score(test_labels, test_preds_binary, average='weighted', zero_division=0)
        precision = precision_score(test_labels, test_preds_binary, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, test_preds_binary, average='weighted')

        metrics_results.append({
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "patch_auc": patch_auc,
            "patient_auc": patient_auc
        })
        #PROBA
        print('METRIQUES',train_idx,'/',test_idx)
        print(metrics_results)
        print('/n')
         

    # Calculate average and standard deviation of metrics
    avg_metrics = {key: np.mean([fold[key] for fold in metrics_results]) for key in metrics_results[0]}
    std_metrics = {key: np.std([fold[key] for fold in metrics_results]) for key in metrics_results[0]}

    print("\nAverage Metrics Across Folds:")
    for key in avg_metrics:
        print(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")

    # Plot the metrics
    metrics_names = list(avg_metrics.keys())
    avg_values = [avg_metrics[m] for m in metrics_names]
    std_values = [std_metrics[m] for m in metrics_names]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_names, avg_values, yerr=std_values, capsize=5, color='skyblue', alpha=0.8)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Average Metrics Across Folds")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_plot_path)  # Save the plot
    print(f"Plot saved to {save_plot_path}")

    
    
# Ruta al archivo Excel con datos y carpeta de imágenes
excel_file_path = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/HP_WSI-CoordAllAnnotatedPatches.xlsx'
data_folder = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/Annotated'

# Cargar datos
patient_ids, labels = load_patient_labels(excel_file_path,154)

# Cargar modelo
model = load_autoencoder_model()

# Validación cruzada del sistema
cross_validate_system(patient_ids, labels, data_folder, model)