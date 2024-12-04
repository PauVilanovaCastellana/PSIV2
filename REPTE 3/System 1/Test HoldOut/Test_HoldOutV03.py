# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:42:22 2024

@author: eduar
"""

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score, roc_curve, auc
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

def load_holdout_labels(csv_path, data_folder):
    """
    Filters the HoldOut dataset to include only patients that exist both in the HoldOut folder and in the CSV,
    excluding patients with the label 'BAIXA'.

    Args:
        csv_path: Path to the CSV file containing patient IDs and labels.
        data_folder: Path to the HoldOut folder containing patient data.

    Returns:
        filtered_ids: List of patient IDs that exist in both the HoldOut folder and the CSV.
        filtered_labels: Corresponding labels for the filtered patient IDs.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Convert the labels in the 'DENSITAT' column to numerical values
    label_mapping = {
        "ALTA": 1,
        "BAIXA": None,  # Exclude 'BAIXA' by mapping it to None
        "NEGATIVA": 0
    }
    df['DENSITAT'] = df['DENSITAT'].map(label_mapping)

    # Drop rows where 'DENSITAT' is None (i.e., exclude 'BAIXA')
    df = df.dropna(subset=['DENSITAT'])

    # Extract patient IDs and labels
    csv_patient_ids = df['CODI'].tolist()
    csv_labels = df['DENSITAT'].tolist()

    # Get the list of patient folders in the HoldOut directory
    holdout_patient_folders = [os.path.basename(folder) for folder in glob.glob(os.path.join(data_folder, "*"))]
    holdout_patient_ids = {folder.split('_')[0] for folder in holdout_patient_folders}

    # Filter the CSV patients to include only those present in the HoldOut folder
    filtered_ids = []
    filtered_labels = []
    for patient_id, label in zip(csv_patient_ids, csv_labels):
        if patient_id in holdout_patient_ids:
            filtered_ids.append(patient_id)
            filtered_labels.append(label)

    # Print debug information
    print(f"Total Patients in HoldOut Folder: {len(holdout_patient_ids)}")
    print(f"Total Patients in CSV: {len(csv_patient_ids)}")
    print(f"Patients Matched: {len(filtered_ids)}")
    from collections import Counter
    print(f"Label Distribution: {Counter(filtered_labels)}")

    return filtered_ids, filtered_labels


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
def evaluate_holdout_set(holdout_ids, holdout_labels, data_folder, model, patch_threshold, patient_threshold,show=True):
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []  # For ROC curve calculation

    filtered_holdout_ids = holdout_ids[:]
    filtered_holdout_labels = holdout_labels[:]

    for patient_id, true_label in zip(filtered_holdout_ids, filtered_holdout_labels):
        F_red_list = process_patient_images(patient_id, data_folder, model)
        total_patches = len(F_red_list)

        if total_patches == 0:
            print(f"No images found for patient {patient_id}.")
            continue

        positive_patches = sum(1 for f_red in F_red_list if f_red > patch_threshold)
        prediction_ratio = positive_patches / total_patches 
        predicted_label = 1 if prediction_ratio >= patient_threshold else 0

        # Append data for evaluation
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(prediction_ratio)  # Use the ratio as probability

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probabilities = np.array(predicted_probabilities)

    if len(true_labels) == 0 or len(predicted_labels) == 0:
        print("Error: No valid data available for evaluation.")
        return None
    
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities, pos_label=1)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    patient_threshold = thresholds[optimal_idx]
    
    if show:
      plt.figure()
      plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
      plt.plot([0, 1], [0, 1], 'k--')
      plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal Patient Th = {patient_threshold:.2f}')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('HoldOut Diagnosis ROC')
      plt.legend(loc="lower right")
      plt.savefig('/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/NOBAIXA_HoldOut_Roc_Curve_prime.png', bbox_inches='tight')
      plt.close()

    # Calculate metrics per class
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

    print("\nMetrics Summary:")
    for cls, cls_metrics in metrics.items():
        print(f"\nClass {cls}:")
        for metric_name, metric_value in cls_metrics.items():
            print(f"  {metric_name.capitalize()}: {metric_value:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title("Confusion Matrix - HoldOut Set")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/NOBAIXA_HoldOut_Confusion_Matrix_Fold_prime.png")
    plt.close()

    # Global metrics
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

    # Plot global metrics
    metrics_names = list(global_metrics.keys())
    avg_values = list(global_metrics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metrics_names, avg_values, color='skyblue', alpha=0.8)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Global Metrics Across HoldOut Set")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/NOBAIXA_HoldOut_Global_Metrics_Fold_prime.png")
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - HoldOut Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/NOBAIXA_HoldOut_ROC_Curve_Fold_prime.png")
    plt.close()

    print("ROC curve saved to /export/fhome/maed01/myvirtualenv/AutoEncoder/Images/Results HoldOut/HoldOut_ROC_Curve_Fold_prime.png")

    return metrics



# Parámetros y ejecución
# Function to count labels and distinct patients
def analyze_holdout_data(holdout_labels, data_folder):
    """
    Analyze the HoldOut data by counting the number of labels and distinct patients.

    Args:
        holdout_labels: List of labels in the HoldOut dataset.
        data_folder: Path to the directory containing HoldOut patient folders.

    Prints:
        - Total label distribution.
        - Number of distinct patients in the HoldOut folder.
    """
    # Count the labels
    from collections import Counter
    label_counts = Counter(holdout_labels)
    print("HoldOut Label Distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples")

    # Count distinct patients by counting directories in the HoldOut folder
    patient_folders = glob.glob(os.path.join(data_folder, "*"))
    distinct_patients = len(patient_folders)
    print(f"Total Distinct Patients in HoldOut Folder: {distinct_patients}")


# Parameters and execution
patch_threshold = 210
patient_threshold = 0.45

# holdout_ids, holdout_labels = load_holdout_labels(csv_path)

csv_path = "/export/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"
data_folder = "/export/fhome/maed/HelicoDataSet/HoldOut/"

filtered_ids, filtered_labels = load_holdout_labels(csv_path, data_folder)
model = load_autoencoder_model()

# Analyze label and patient counts
analyze_holdout_data(filtered_labels, data_folder)

# Evaluate HoldOut set
metrics = evaluate_holdout_set(filtered_ids, filtered_labels, data_folder, model, patch_threshold, patient_threshold)
