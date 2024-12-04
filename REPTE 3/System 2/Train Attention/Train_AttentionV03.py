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

    # Create a mapping dictionary to convert 'DENSITAT' values
    densitat_map = {
        'NEGATIVA': 0,
        'BAIXA': 1,
        'ALTA': 1
    }

    # Verify that all values in 'DENSITAT' are valid
    if not all(dens in densitat_map for dens in df['DENSITAT'].unique()):
        raise ValueError("Unexpected values found in the 'DENSITAT' column. "
                         "Ensure it contains only 'NEGATIVA', 'BAIXA', or 'ALTA'.")

    # Map 'DENSITAT' values using the dictionary
    df['DENSITAT'] = df['DENSITAT'].map(densitat_map)

    # Count unique patients
    unique_patients = len(df['CODI'].unique())
    print(f"Number of unique patients: {unique_patients}")

    # Count patients by class
    patient_counts_by_class = df['DENSITAT'].value_counts().to_dict()
    print(f"Patient counts by class: {patient_counts_by_class}")

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
        # x: [max_features, feature_dim] (las características del paciente)
        Z, attention_weights = self.attention(x)  # Z: [attention_branches, feature_dim]
        Z = Z.view(Z.size(0), -1)  # Flatten the attention output -> [1, feature_dim * attention_branches]
        output = self.classifier(Z)  # Pasar por la red clasificadora
        return output, attention_weights


def train_attention_model(train_loader, attention_model, epochs=5, learning_rate=1e-4, save_path="attention_model.pth", accumulation_steps=309):
    criterion = nn.CrossEntropyLoss()  # Pérdida para clasificación binaria
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=learning_rate)

    # Ajustar el scheduler para reducir la tasa de aprendizaje
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    best_loss = float('inf')  # Inicializa la mejor pérdida

    for epoch in range(epochs):
        attention_model.train()  # Modo entrenamiento
        epoch_loss = 0.0
        loss = 0.0
        optimizer.zero_grad()  # Inicializar gradientes acumulados

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.squeeze(0).to(device)  # features -> [max_features, feature_dim]
            labels = labels.to(device).view(-1).long()  # Asegurar etiquetas correctas -> [1]

            valid_features = features[~torch.all(features == 0, dim=1)]  # Filtrar tensores de relleno
            outputs, attention_weights = attention_model(valid_features)  # Salida del modelo
            patient_output = outputs.mean(dim=0, keepdim=True)  # Promediar las salidas -> [1, output_classes]

            losspat = criterion(patient_output, labels)
            loss += losspat / accumulation_steps

            epoch_loss += losspat.item()  # Acumulación de la pérdida de la época

            # Backpropagation y optimización
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                loss.backward()  # Acumular gradientes
                optimizer.step()  # Actualizar parámetros
                optimizer.zero_grad()  # Reiniciar gradientes acumulados
                loss = 0

        # Imprimir la pérdida promedio por época
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}")

        # Ajustar la tasa de aprendizaje usando el scheduler
        scheduler.step(epoch_loss)

        # Guardar el modelo si la pérdida mejora
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(attention_model.state_dict(), save_path)
            print(f"Modelo mejorado guardado en {save_path}")

    print("Entrenamiento completado.")







# Ruta de datos
data_folder = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/Cropped'
file_path = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/PatientDiagnosis.csv'

# Cargar etiquetas y datos
patient_ids, labels = load_patient_labels(file_path)
encoder = load_encoder_model()
train_dataset = PatientDataset(patient_ids, labels, data_folder, encoder)
print(f"Número de pacientes en el dataset: {len(train_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Modelo de atención
attention_model = AttentionPatientDiagnosis(feature_dim=64, decom_space=128, attention_branches=1, output_classes=2).to(device)

# Entrenar y  guardar el modelo
train_attention_model(train_dataloader, attention_model, epochs=10, learning_rate=1e-4, save_path="attention_model.pth", accumulation_steps=309)  # Acumular gradientes de 4 pacientes)
total_params = sum(p.numel() for p in attention_model.parameters())
print(f"Total parámetros en el modelo: {total_params}")
