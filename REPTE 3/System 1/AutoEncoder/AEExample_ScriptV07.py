import sys
import os
import pickle
import numpy as np
import pandas as pd
import cv2  
import glob
import random
import torch
from torch.utils.data import DataLoader, Dataset
import gc
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from Models.weights_init import weights_init_xavier

def AEConfigs(Config):
    net_paramsEnc = {}
    net_paramsDec = {}
    inputmodule_paramsDec = {}
    
    # Aumentamos el drop_rate para reducir la capacidad de reconstrucción detallada
    net_paramsEnc['drop_rate'] = 0.4
    net_paramsDec['drop_rate'] = 0.4
    
    if Config == '1':
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec
49430
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
cropped_images = cropped_images[:3000]

# Preprocesamiento de imágenes
patches = []
for image_path in cropped_images:
    image = cv2.imread(image_path)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (256, 256))
    normalized_image = resized_image / 255.0
    patches.append(normalized_image)

# Crear el dataset
patches = np.array(patches, dtype=np.float32).transpose(0, 3, 1, 2)
train_dataset = Standard_Dataset(patches)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Definir el modelo
Config = '1'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
model.apply(weights_init_xavier)

# Configurar el entrenamiento
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # LR más alto
criterion = torch.nn.MSELoss()

# Entrenamiento con menos épocas y sin ajuste de lr
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        batch = batch.float()
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_loader)}')

# Guardar el modelo entrenado
model_path = 'autoencoder_model_25epochs_lr002.pth'
torch.save(model.state_dict(), model_path)

gc.collect()
torch.cuda.empty_cache()
