import sys
import os
import numpy as np
import pandas as pd
import cv2  
import glob
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gc
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from Models.weights_init import weights_init_xavier, weights_init_kaiming

# Configurar el dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Parámetros del modelo y los datos
inputmodule_paramsEnc = {'num_input_channels': 3}
data_folder = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/Cropped'
file_path = r'/export/fhome/maed01/HelicoDataSet/CrossValidation/PatientDiagnosis.csv'

# Cargar datos
try:
    patient_diagnosis = pd.read_csv(file_path)
except FileNotFoundError:
    sys.exit(f"Error: No se encontró el archivo {file_path}")

# Filtrar imágenes con diagnóstico negativo
negative_diagnosis = patient_diagnosis[patient_diagnosis['DENSITAT'] == 'NEGATIVA']
negative_patient_ids = negative_diagnosis['CODI'].values

cropped_images = []
for patient_id in negative_patient_ids:
    patient_folders = glob.glob(os.path.join(data_folder, f"{patient_id}_*"))
    for patient_folder in patient_folders:
        images = glob.glob(os.path.join(patient_folder, '*.png'))
        cropped_images.extend(images)

print(f"Total de imágenes cargadas: {len(cropped_images)}")
random.shuffle(cropped_images)
cropped_images = cropped_images[:10000]

# Preprocesamiento de imágenes en RGB
patches = []
for image_path in cropped_images:
    image = cv2.imread(image_path)
    if image is None:
        continue
    resized_image = cv2.resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    patches.append(normalized_image)

# Crear el dataset
patches = np.array(patches, dtype=np.float32).transpose(0, 3, 1, 2)  # [N, C, H, W]
train_dataset = Standard_Dataset(patches)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # batch_size aumentado a 5000

# Definir el modelo y moverlo a la GPU
Config = '1'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
model.apply(weights_init_kaiming)  # Usar inicialización de Kaiming
model.to(device)

# Configurar el entrenamiento
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Reducción de LR
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reducir LR cada 10 épocas
criterion = torch.nn.MSELoss()

# Configuración de Early Stopping
early_stop_patience = 5
best_loss = float('inf')
epochs_no_improve = 0

# Entrenamiento con early stopping y ajuste de LR
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.float().to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Ajustar la tasa de aprendizaje
    scheduler.step()
    
    # Calcular la pérdida media para esta época
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]}')
    
    # Early Stopping: Comprobar si la pérdida mejora
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        epochs_no_improve = 0
        # Guardar el modelo con la mejor pérdida
        model_path = 'autoencoder_model_V11_best.pth'
        torch.save(model.state_dict(), model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered. Training terminated.")
            break

# Limpiar memoria
gc.collect()
torch.cuda.empty_cache()
