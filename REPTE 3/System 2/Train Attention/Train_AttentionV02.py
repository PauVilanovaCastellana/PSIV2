# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:56:55 2024

@author: eduar
"""

# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Models.AEmodels import AutoEncoderCNN
from AttentionUnits import GatedAttention

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Función para cargar etiquetas del conjunto de datos
def load_patient_labels(file_path):
    df = pd.read_csv(file_path)
    if 'CODI' not in df.columns or 'DENSITAT' not in df.columns:
        raise KeyError("The CSV file must contain the columns 'CODI' and 'DENSITAT'.")

    # Mapeo de valores de 'DENSITAT'
    densitat_map = {
        'NEGATIVA': 0,
        'BAIXA': 1,
        'ALTA': 1
    }

    # Validación de valores en 'DENSITAT'
    if not all(dens in densitat_map for dens in df['DENSITAT'].unique()):
        raise ValueError("Unexpected values found in the 'DENSITAT' column. "
                         "Ensure it contains only 'NEGATIVA', 'BAIXA', or 'ALTA'.")

    # Mapeo de etiquetas
    df['DENSITAT'] = df['DENSITAT'].map(densitat_map)

    patient_ids = df['CODI'].tolist()
    labels = df['DENSITAT'].tolist()
    return patient_ids, labels

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

# Función para cargar solo el encoder del autoencoder con reducción de dimensiones
def load_encoder_model():
    Config = '1'
    inputmodule_paramsEnc = {'num_input_channels': 3}
    net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config, inputmodule_paramsEnc)
    autoencoder = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    autoencoder.load_state_dict(torch.load('best_autoencoder_model.pth', map_location=device))
    autoencoder.eval()

    # Crear un modelo con reducción de dimensiones
    encoder = autoencoder.encoder
    encoder_dim_reduction = nn.Sequential(
        encoder,
        nn.Flatten(),
        nn.Linear(64 * 64 * 64, 64)  # Reducción a 64 dimensiones
    )
    return encoder_dim_reduction.to(device)

# Dataset personalizado para cargar datos de pacientes
class PatientDataset(Dataset):
    def __init__(self, patient_ids, labels, data_folder, encoder, max_features=100):
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_folder = data_folder
        self.encoder = encoder
        self.max_features = max_features  # Máximo número de características por paciente
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
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convertir etiqueta en tensor
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

        # Ajustar el tamaño de las características
        if len(features) == 0:
            features = torch.zeros((self.max_features, 64), device=device)
        else:
            features = torch.stack(features)
            if features.size(0) > self.max_features:
                features = features[:self.max_features]  # Recortar
            elif features.size(0) < self.max_features:
                padding = torch.zeros((self.max_features - features.size(0), 64), device=device)
                features = torch.cat([features, padding], dim=0)

        return features, label

# Modelo de atención para la clasificación de pacientes
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

# Entrenamiento del modelo de atención
def train_attention_model(dataloader, attention_model, epochs=50, learning_rate=1e-3, save_path="attention_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=learning_rate)
    attention_model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)  # Mover ambos al dispositivo

            optimizer.zero_grad()
            # Procesar las características en el modelo
            batch_outputs = []
            for patient_features in features:  # Procesar cada paciente individualmente
                valid_features = patient_features[~torch.all(patient_features == 0, dim=1)]  # Filtrar tensores de relleno
                outputs, _ = attention_model(valid_features)
                patient_output = outputs.mean(dim=0, keepdim=True)  # Promediar las salidas del paciente
                batch_outputs.append(patient_output)

            batch_outputs = torch.cat(batch_outputs, dim=0)
            loss = criterion(batch_outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(attention_model.state_dict(), save_path)
    print(f"Modelo de atención guardado en {save_path}")

# Ruta de datos
data_folder = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/Cropped'
file_path = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/PatientDiagnosis.csv'

# Cargar etiquetas y datos
patient_ids, labels = load_patient_labels(file_path)
encoder = load_encoder_model()
dataset = PatientDataset(patient_ids, labels, data_folder, encoder)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Modelo de atención
attention_model = AttentionPatientDiagnosis(feature_dim=64, decom_space=128, attention_branches=1, output_classes=2).to(device)

# Entrenar y guardar modelo
train_attention_model(dataloader, attention_model, epochs=20, learning_rate=1e-3, save_path="attention_model.pth")
