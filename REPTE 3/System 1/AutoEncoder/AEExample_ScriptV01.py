# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides: 
    1. Split into train and test steps 
    2. Save trained models and any intermediate result input of the next step
    
@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: https://arxiv.org/abs/2309.16053 

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

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch


## Own Functions
from Models.AEmodels import AutoEncoderCNN

## Others
from torch.utils.data import Dataset, DataLoader



def AEConfigs(Config):
    # Inicializar las variables de configuración antes de usarlas
    net_paramsEnc = {}  # Inicializa el diccionario para la red de codificación
    net_paramsDec = {}  # Inicializa el diccionario para la red de decodificación
    inputmodule_paramsDec = {}  # Inicializa el diccionario para el módulo de entrada
    
    # Aquí agregamos 'drop_rate' al diccionario net_paramsEnc
    net_paramsEnc['drop_rate'] = 0.5  # Ejemplo de valor 
    net_paramsDec['drop_rate'] = 0.5  # Ejemplo de valor 
    
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

# 0.1 NETWORK TRAINING PARAMS
# (Aquí puedes agregar cualquier parámetro específico para la configuración del entrenamiento)

# 0.2 FOLDERS
data_folder = '/export/fhome/maed01/HelicoDataSet/CrossValidation/Cropped'

# 1. LOAD DATA
# 1.1 Cargar el fichero PatientDiagnosis
file_path = '/export/fhome/maed01/HelicoDataSet/CrossValidation/PatientDiagnosis.csv'
patient_diagnosis = pd.read_csv(file_path)

# Filtrar las imágenes con diagnóstico negativo (Healthy patients)
negative_diagnosis = patient_diagnosis[patient_diagnosis['DENSITAT'] == 'NEGATIVA']
negative_patient_ids = negative_diagnosis['CODI'].values  # Lista de IDs de pacientes con diagnóstico negativo

# 1.2 Obtener las imágenes de la carpeta Cropped que corresponden a estos pacientes
cropped_images = []
for patient_id in negative_patient_ids:
    # Buscar carpetas que empiecen con el ID del paciente seguido de un guion bajo y un número
    patient_folders = glob.glob(os.path.join(data_folder, f"{patient_id}_*"))
    
    for patient_folder in patient_folders:
        images = glob.glob(os.path.join(patient_folder, '*.png'))  # Obtener todas las imágenes .png
        cropped_images.extend(images)

print(f"Total de imágenes cargadas: {len(cropped_images)}")

cropped_images = cropped_images[:100]
#### 2. DATA SPLITTING INTO INDEPENDENT SETS
# Aquí puedes definir cómo dividir los datos en sets de entrenamiento y test, si es necesario

#### 3. LOAD PATCHES
# Usaremos OpenCV para leer las imágenes (parches) y las convertiremos en escala de grises
patches = []  # Lista de parches de las imágenes

for image_path in cropped_images:
    image = cv2.imread(image_path)  
    resized_image = cv2.resize(image, (256, 256))  
    patches.append(resized_image)

cv2.imshow('Primer parche', patches[0])  # Muestra la primera imagen en la lista
cv2.waitKey(0)  # Espera a que el usuario presione una tecla
cv2.destroyAllWindows() 

# 4. AE TRAINING

# CONFIG1
Config = '1'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

# Inicializar el modelo AutoEncoder
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                       inputmodule_paramsDec, net_paramsDec)

# 3. LOAD PATCHES
# Usaremos OpenCV para leer las imágenes (parches) y las convertiremos en RGB (y opcionalmente normalizamos)

patches = []  # Lista de parches de las imágenes
for image_path in cropped_images:
    image = cv2.imread(image_path)  # Usamos OpenCV para leer la imagen en color (por defecto en BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
    image_rgb = image_rgb / 255.0  # Normalizar la imagen entre 0 y 1
    patches.append(image_rgb)

# Convertir las imágenes a tensor de PyTorch
class PatchesDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Convertir la imagen a tensor
        image = self.patches[idx]
        image_tensor = torch.tensor(image, dtype=torch.float32)  # Convertir a tensor flotante
        image_tensor = image_tensor.permute(2, 0, 1)  # Cambiar la forma a (C, H, W) para PyTorch
        return image_tensor

# Crear el dataset y dataloader
patches_dataset = PatchesDataset(patches)
train_loader = DataLoader(patches_dataset, batch_size=32, shuffle=True)

# Optimización y función de pérdida
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()  # Usamos MSE como función de pérdida

num_epochs = 10  # Número de épocas de entrenamiento
for epoch in range(num_epochs):
    model.train()  # Poner el modelo en modo de entrenamiento
    for batch in train_loader:
        optimizer.zero_grad()  # Resetear los gradientes
        output = model(batch)  # Forward pass
        loss = criterion(output, batch)  # Calcular la pérdida

        loss.backward()  # Backward pass
        optimizer.step()  # Actualizar los pesos

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Free GPU Memory After Training
gc.collect()
torch.cuda.empty_cache()
