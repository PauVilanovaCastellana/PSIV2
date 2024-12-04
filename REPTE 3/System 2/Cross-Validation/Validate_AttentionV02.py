import os
import glob
import pandas as pd
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
from Models.AEmodels import AutoEncoderCNN
from AttentionUnits import GatedAttention
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Función para cargar etiquetas del conjunto de datos
def load_patient_labels(file_path, k):
    df = pd.read_excel(file_path)
    if 'Pat_ID' not in df.columns or 'Presence' not in df.columns:
        raise KeyError("El archivo Excel debe contener las columnas 'Pat_ID' y 'Presence'.")
    
    patient_ids = df['Pat_ID'].tolist()
    labels = df['Presence'].tolist()
    labels = [0 if label == -1 else label for label in labels]
    
    unique_patients = list(set(patient_ids))[:k]
    k_patient_ids = []
    k_labels = []
    for patient in unique_patients:
        for pid, label in zip(patient_ids, labels):
            if patient == pid:
                k_patient_ids.append(pid)
                k_labels.append(label)
    return k_patient_ids, k_labels

# Configuración del Autoencoder
def AEConfigs(Config, inputmodule_paramsEnc):
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

# Cargar el modelo de Autoencoder preentrenado
def load_encoder_model():
    Config = '1'
    inputmodule_paramsEnc = {'num_input_channels': 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config, inputmodule_paramsEnc)
    autoencoder = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    autoencoder.load_state_dict(torch.load('best_autoencoder_model.pth', map_location=device))
    autoencoder.eval()
    encoder = autoencoder.encoder
    encoder_dim_reduction = nn.Sequential(
        encoder,
        nn.Flatten(),
        nn.Linear(64 * 64 * 64, 64)
    )
    return encoder_dim_reduction.to(device)

# Dataset personalizado
class PatientDataset(Dataset):
    def __init__(self, patient_ids, labels, data_folder, encoder, max_features=100):
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_folder = data_folder
        self.encoder = encoder
        self.max_features = max_features
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        patient_folder = glob.glob(os.path.join(self.data_folder, f"{patient_id}_*"))

        features = []
        for folder in patient_folder:
            image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        feature = self.encoder(img.to(device))
                        features.append(feature.squeeze())

        if len(features) == 0:
            features = torch.zeros((self.max_features, 64), device=device)
        else:
            features = torch.stack(features)
            if features.size(0) > self.max_features:
                features = features[:self.max_features]
            elif features.size(0) < self.max_features:
                padding = torch.zeros((self.max_features - features.size(0), 64), device=device)
                features = torch.cat([features, padding], dim=0)

        return features, label

# Modelo de atención
class AttentionPatientDiagnosis(nn.Module):
    def __init__(self, feature_dim=64, decom_space=128, attention_branches=1, output_classes=2):
        super(AttentionPatientDiagnosis, self).__init__()
        self.attention = GatedAttention({
            'in_features': feature_dim,
            'decom_space': decom_space,
            'ATTENTION_BRANCHES': attention_branches
        })
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * attention_branches, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_classes)
        )

    def forward(self, x):
        Z, attention_weights = self.attention(x)
        Z = Z.view(Z.size(0), -1)
        output = self.classifier(Z)
        return output, attention_weights

def evaluate_patient_classification_crossval(patient_ids, labels, dataset, model, n_splits=5, save_path="Evaluation_Metrics_CV.csv"):
    """
    Realiza validación cruzada basada en grupos (GroupKFold) y guarda las métricas de evaluación.

    Args:
        patient_ids (list): Lista de IDs de pacientes (para grupos).
        labels (list): Etiquetas de los pacientes.
        dataset (Dataset): Dataset de pacientes.
        model (nn.Module): Modelo de atención.
        n_splits (int): Número de splits para GroupKFold.
        save_path (str): Ruta para guardar las métricas en un archivo CSV.
    """
    gkf = GroupKFold(n_splits=n_splits)
    metrics_per_fold = []
    all_true_labels = []  # Para acumular las etiquetas verdaderas de todos los folds
    all_predicted_labels = []  # Para acumular las predicciones de todos los folds

    for fold, (train_idx, test_idx) in enumerate(gkf.split(patient_ids, labels, groups=patient_ids), start=1):
        print(f"\nFold {fold}/{n_splits}")

        # Dividir el dataset en entrenamiento y validación
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Crear DataLoaders
        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

        # Evaluar en el fold actual
        true_labels = []
        predicted_labels = []

        model.eval()  # Modo evaluación
        with torch.no_grad():
            for features, labels in test_loader:
                for patient_features in features:
                    valid_features = patient_features[~torch.all(patient_features == 0, dim=1)]
                    outputs, _ = model(valid_features.to(device))
                    patient_output = outputs.mean(dim=0)
                    prediction = torch.argmax(patient_output)
                    predicted_labels.append(prediction.item())
                
                true_labels.extend(labels.numpy())

        # Acumular resultados globales
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)

        # Calcular métricas por clase para este fold
        accuracy = accuracy_score(true_labels, predicted_labels)
        all_classes = sorted(set(true_labels))
        precision = precision_score(true_labels, predicted_labels, labels=all_classes, average=None, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, labels=all_classes, average=None, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, labels=all_classes, average=None, zero_division=0)

        fold_metrics = {
            "Fold": fold,
            "Accuracy": accuracy,
        }
        for cls in range(max(labels) + 1):  # Iterar sobre todas las clases posibles
            fold_metrics[f"Precision Class {cls}"] = precision[cls] if cls in all_classes else 0
            fold_metrics[f"Recall Class {cls}"] = recall[cls] if cls in all_classes else 0
            fold_metrics[f"F1 Class {cls}"] = f1[cls] if cls in all_classes else 0

        # Guardar métricas del fold
        metrics_per_fold.append(fold_metrics)

        print(f"Fold {fold} Metrics:")
        print(fold_metrics)

        # Visualizar la matriz de confusión
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"Confusion_Matrix_Fold_{fold}.png")
        plt.close()
        print(f"Matriz de confusión del fold {fold} guardada como 'Confusion_Matrix_Fold_{fold}.png'")

    # Guardar las métricas de todos los folds en un archivo CSV
    metrics_df = pd.DataFrame(metrics_per_fold)
    metrics_df.to_csv(save_path, index=False)
    print(f"Métricas de validación cruzada guardadas en {save_path}")

    # Calcular métricas promedio y desviación estándar por clase
    avg_metrics = metrics_df.mean(numeric_only=True)
    std_metrics = metrics_df.std(numeric_only=True)

    print("\nPromedios de métricas por clase:")
    print(avg_metrics)
    print("\nDesviación estándar de métricas por clase:")
    print(std_metrics)

    # Calcular métricas globales (macro-promedio)
    global_precision = precision_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    global_recall = recall_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    global_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    
    print("\nMétricas globales (macro-promedio):")
    print(f"  Precision: {global_precision}")
    print(f"  Recall: {global_recall}")
    print(f"  F1: {global_f1}")
    
    # Crear gráfico de métricas globales agregadas (sin accuracy)
    metric_names = ["Precision", "Recall", "F1"]
    avg_values = [global_precision, global_recall, global_f1]
    
    # Ahora 'avg_values' tiene la forma correcta: (3,)
    plt.figure(figsize=(8, 6))
    plt.bar(metric_names, avg_values, color='skyblue', alpha=0.8)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Global Metrics (Precision, Recall, F1)")
    plt.tight_layout()
    plt.savefig("Global_Metrics_Barplot.png")
    plt.close()
    print("Gráfica de métricas globales guardada como 'Global_Metrics_Barplot.png'")




# Rutas y parámetros
excel_file_path = '/export/fhome/maed01/HelicoDataSet/CrossValidation/HP_WSI-CoordAllAnnotatedPatches.xlsx'
data_folder = '/export/fhome/maed01/HelicoDataSet/CrossValidation/Annotated'
attention_model_path = 'attention_model.pth'
k = 154

# Cargar datos
k_patient_ids, k_labels = load_patient_labels(excel_file_path, k)
encoder = load_encoder_model()
dataset = PatientDataset(k_patient_ids, k_labels, data_folder, encoder)

# Cargar modelo de atención
attention_model = AttentionPatientDiagnosis(feature_dim=64, decom_space=128, attention_branches=1, output_classes=2)
attention_model.load_state_dict(torch.load(attention_model_path, map_location=device))
attention_model.to(device)

# Evaluar con validación cruzada basada en grupos
evaluate_patient_classification_crossval(k_patient_ids, k_labels, dataset, attention_model, n_splits=20, save_path="Metrics_CrossValidation.csv")




