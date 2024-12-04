# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides: 
    1. Split into train and test steps 
    2. Save trained models and any intermediate result input of the next step
"""
# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob
import cv2  # Usaremos OpenCV para la lectura de imágenes
import random

# Torch Libraries
from torch.utils.data import DataLoader, Dataset
import gc
import torch

# Own Functions
from Models.AEmodels import AutoEncoderCNN


def AEConfigs(Config):
    # Inicializar las variables de configuración antes de usarlas
    net_paramsEnc = {}  
    net_paramsDec = {}  
    inputmodule_paramsDec = {}  
    
    # Configuración de drop_rate
    net_paramsEnc['drop_rate'] = 0.5  
    net_paramsDec['drop_rate'] = 0.5  
    
    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    elif Config == '3':  
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]
    
    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc = {}
inputmodule_paramsEnc['num_input_channels'] = 3  # Suponiendo que las imágenes tienen 3 canales (RGB)

# 0.2 FOLDERS
data_folder = '/export/fhome/maed01/HelicoDataSet/CrossValidation/Cropped'
file_path = '/export/fhome/maed01/HelicoDataSet/CrossValidation/PatientDiagnosis.csv'

# 1. LOAD DATA
# 1.1 Cargar el fichero PatientDiagnosis
try:
    patient_diagnosis = pd.read_csv(file_path)
except FileNotFoundError:
    sys.exit(f"Error: No se encontro el archivo {file_path}")


# Filtrar las imágenes con diagnóstico negativo (Healthy patients)
negative_diagnosis = patient_diagnosis[patient_diagnosis['DENSITAT'] == 'NEGATIVA']
negative_patient_ids = negative_diagnosis['CODI'].values  

# 1.2 Obtener las imágenes de la carpeta Cropped que corresponden a estos pacientes
cropped_images = []
for patient_id in negative_patient_ids:
    # Para cada paciente con diagnóstico negativo, buscar carpetas que correspondan a su código
    patient_folders = glob.glob(os.path.join(data_folder, f"{patient_id}_*"))
    
    for patient_folder in patient_folders:
        images = glob.glob(os.path.join(patient_folder, '*.png'))
        cropped_images.extend(images)

if not cropped_images:
    sys.exit("Error: No se encontraron imagenes para pacientes sanos en la carpeta especificada.")

print(f"Total de imagenes cargadas: {len(cropped_images)}")

random.shuffle(cropped_images)
cropped_images = cropped_images[:1500]

# 2. LOAD PATCHES
patches = []
for image_path in cropped_images:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        continue
    resized_image = cv2.resize(image, (256, 256))
    patches.append(resized_image)

# Verificar que la lista `patches` no esté vacía antes de mostrar una imagen
if patches:
    cv2.imshow('Primer parche', patches[0])  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: No se pudo cargar ninguna imagen.")

# 4. AE TRAINING
Config = '2'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)

# Convertir imágenes a tensor y normalizar
patches = []
for image_path in cropped_images:
    image = cv2.imread(image_path)
    if image is None:
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    patches.append(image_rgb)

class PatchesDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        image = self.patches[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

patches_dataset = PatchesDataset(patches)
train_loader = DataLoader(patches_dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

gc.collect()
torch.cuda.empty_cache()


gc.collect()
torch.cuda.empty_cache()