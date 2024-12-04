# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Models.AEmodels import AutoEncoderCNN
from AttentionUnits import GatedAttention
import torch.nn as nn


# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Función para cargar etiquetas del conjunto de datos
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
        z, attention_weights = self.attention(x)
        z = z.view(z.size(0), -1)
        output = self.classifier(z)
        return output, attention_weights

# Validación cruzada
def cross_validate(patient_ids, labels, data_folder, encoder, n_splits=5, save_path="/Images/Cross_Validation_ResultsV01.csv"):
    model = AttentionPatientDiagnosis(feature_dim=64, decom_space=128, attention_branches=1, output_classes=2)
    model.load_state_dict(torch.load("attention_model.pth", map_location=device))
    model.to(device)
    model.eval()

    gkf = GroupKFold(n_splits=n_splits)
    metrics_data = []
    overall_conf_matrix = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(patient_ids, labels, groups=patient_ids), start=1):
        test_ids = [patient_ids[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        test_dataset = PatientDataset(test_ids, test_labels, data_folder, encoder)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        true_labels, predicted_labels = [], []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                for patient_features in features:
                    valid_features = patient_features[~torch.all(patient_features == 0, dim=1)]
                    outputs, _ = model(valid_features)
                    patient_output = outputs.mean(dim=0)
                    predicted_labels.append(torch.argmax(patient_output).item())
                true_labels.extend(labels.cpu().numpy())

        # Guardar y acumular la matriz de confusión
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
        overall_conf_matrix += conf_matrix

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f"Confusion Matrix for Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"/Images/Confusion_Matrix_Fold_{fold}_V01.png")
        plt.close()
        print(f"/Images/Confusion matrix for fold {fold} saved.")

        # Calcular métricas
        precision = precision_score(true_labels, predicted_labels, average=None)
        recall = recall_score(true_labels, predicted_labels, average=None)
        f1 = f1_score(true_labels, predicted_labels, average=None)

        for cls in range(2):
            metrics_data.append({
                "fold": fold,
                "class": cls,
                "precision": precision[cls],
                "recall": recall[cls],
                "f1_score": f1[cls]
            })

        print(f"Fold {fold} complete.")

    # Guardar métricas en un CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}.")

    # Imprimir métricas por clase
    metrics_summary = metrics_df.groupby("class")[["precision", "recall", "f1_score"]].mean()
    metrics_std = metrics_df.groupby("class")[["precision", "recall", "f1_score"]].std()

    print("\nMetrics per class:")
    for cls in metrics_summary.index:
        print(f"Class {cls}:")
        print(f"  Precision: {metrics_summary.loc[cls, 'precision']:.4f} ± {metrics_std.loc[cls, 'precision']:.4f}")
        print(f"  Recall: {metrics_summary.loc[cls, 'recall']:.4f} ± {metrics_std.loc[cls, 'recall']:.4f}")
        print(f"  F1 Score: {metrics_summary.loc[cls, 'f1_score']:.4f} ± {metrics_std.loc[cls, 'f1_score']:.4f}")

    # Guardar la matriz de confusión acumulativa
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("/Images/Overall_Confusion_Matrix_V01.png")
    plt.close()
    print("/Images/Overall confusion matrix saved.")

    # Crear gráfico de métricas promedio y desviación estándar
    metrics_summary["precision_std"] = metrics_std["precision"]
    metrics_summary["recall_std"] = metrics_std["recall"]
    metrics_summary["f1_score_std"] = metrics_std["f1_score"]

    metrics_summary.reset_index(inplace=True)
    metrics_names = ["precision", "recall", "f1_score"]
    x_labels = [f"Class {cls} - {metric}" for cls in metrics_summary["class"] for metric in metrics_names]
    avg_values = [metrics_summary.loc[cls, metric] for cls in range(len(metrics_summary)) for metric in metrics_names]
    std_values = [metrics_summary.loc[cls, f"{metric}_std"] for cls in range(len(metrics_summary)) for metric in metrics_names]

    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, avg_values, yerr=std_values, capsize=5, color='skyblue', alpha=0.8)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Cross-Validation Metrics Summary")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("/Images/Metrics_Summary_V01.png")
    plt.close()
    print("/Images/Metrics summary plot saved.")
    
    

# Configuración de datos
excel_file_path = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/HP_WSI-CoordAllAnnotatedPatches.xlsx'
data_folder = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/Annotated'

patient_ids, labels = load_patient_labels(file_path)
encoder = torch.load("encoder_model.pth", map_location=device)

cross_validate(patient_ids, labels, data_folder, encoder, n_splits=5, save_path="cross_validation_results.csv")
