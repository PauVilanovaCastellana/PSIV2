import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.init as init
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta a los datos
DATASET_PATH = "/export/fhome/maed/EpilepsyDataSet/"

# Dataset personalizado para cargar ventanas EEG y etiquetas
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eeg_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

# Función para cargar datos por paciente con balance de clases
def load_data_by_patient(dataset_path):
    patient_data = {}

    for file in os.listdir(dataset_path):
        if file.endswith(".npz"):
            patient_id = file.split("_")[0]
            npz_path = os.path.join(dataset_path, file)
            data = np.load(npz_path, allow_pickle=True)["EEG_win"]

            file_parts = file.split("_")
            base_name = "_".join(file_parts[:2])
            suffix_number = file_parts[-1].replace(".npz", "")
            parquet_filename = f"{base_name}_metadata_{suffix_number}.parquet"
            parquet_path = os.path.join(dataset_path, parquet_filename)
            metadata = pd.read_parquet(parquet_path)

            labels = metadata["class"].values
            class_0_indices = np.where(labels == 0)[0]
            class_1_indices = np.where(labels == 1)[0]

            n_samples_per_class = min(len(class_0_indices), len(class_1_indices), 5000)

            selected_class_0_indices = np.random.choice(class_0_indices, n_samples_per_class, replace=False)
            selected_class_1_indices = np.random.choice(class_1_indices, n_samples_per_class, replace=False)

            selected_indices = np.concatenate([selected_class_0_indices, selected_class_1_indices])
            np.random.shuffle(selected_indices)

            filtered_data = data[selected_indices]
            filtered_labels = labels[selected_indices]

            if patient_id not in patient_data:
                patient_data[patient_id] = {"data": [], "labels": []}
            patient_data[patient_id]["data"].append(filtered_data)
            patient_data[patient_id]["labels"].append(filtered_labels)

    for patient_id in patient_data:
        patient_data[patient_id]["data"] = np.concatenate(patient_data[patient_id]["data"], axis=0)
        patient_data[patient_id]["labels"] = np.concatenate(patient_data[patient_id]["labels"], axis=0)

    return patient_data

# Preparar datos para entrenamiento y prueba
def prepare_data(patient_ids, patient_data):
    eeg_data = []
    labels = []
    for patient_id in patient_ids:
        eeg_data.append(patient_data[patient_id]["data"])
        labels.append(patient_data[patient_id]["labels"])
    eeg_data = np.concatenate(eeg_data, axis=0).astype(np.float32)  # Asegurar np.float32
    labels = np.concatenate(labels, axis=0).astype(np.float32)  # Asegurar np.float32
    return eeg_data, labels

# Modelo CNN con atención y regularización
class EEGAttentionCNN(nn.Module):
    def __init__(self, num_channels=21):
        super(EEGAttentionCNN, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))

        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))

        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))

        # Mecanismo de atención espacial
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),  # Reduce a un mapa espacial
            nn.Sigmoid()
        )

        # Mecanismo de atención por canal (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 8, kernel_size=1),  # Squeeze
            nn.ReLU(),
            nn.Conv2d(128 // 8, 128, kernel_size=1),  # Excitation
            nn.Sigmoid()
        )

        # Capas totalmente conectadas (dinámica)
        self.fc1 = None  # Placeholder for dynamic initialization
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.9)  
        self.sigmoid = nn.Sigmoid()

    def _initialize_fc1(self, x):
        """Helper to initialize fc1 dynamically based on input size."""
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 256).to(x.device)

    def forward(self, x):
        # Extracción inicial
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Extracción intermedia
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Extracción final
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Atención espacial
        spatial_att = self.spatial_attention(x)  # [batch, 1, h, w]
        x = x * spatial_att

        # Atención por canal
        channel_att = self.channel_attention(x)  # [batch, c, 1, 1]
        x = x * channel_att

        # Aplanado para las capas FC
        x = x.view(x.size(0), -1)

        # Dinámicamente inicializar fc1
        if self.fc1 is None:
            self._initialize_fc1(x)

        # Clasificación
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Función de entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Validación cruzada Leave-One-Patient-Out
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Cargando datos por paciente...")
    patient_data = load_data_by_patient(DATASET_PATH)
    patient_ids = list(patient_data.keys())

    all_train_losses = []
    all_val_losses = []
    all_confusion_matrices = []
    all_classification_reports = []

    fold = 1
    for test_patient in patient_ids:
        print(f"Fold {fold}: Test Patient {test_patient}")

        train_patients = [p for p in patient_ids if p != test_patient]
        
        # Preparar datos para entrenamiento y prueba
        print("Preparando datos para entrenamiento y prueba...")
        X_train, y_train = prepare_data(train_patients, patient_data)
        X_test, y_test = prepare_data([test_patient], patient_data)
    
        # Verificar y convertir a np.float32
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
    
        # Asegurar que las dimensiones sean correctas antes de la normalización
        if X_train.ndim == 3:  # [NWinPatID, 21, 128]
            # Reemplazar valores no válidos (NaN o inf) con 0
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
            # Normalizar los datos con protección contra std=0
            std_train = np.std(X_train, axis=(1, 2), keepdims=True)
            std_train[std_train == 0] = 1e-6  # Reemplazar std=0 por un valor pequeño
            X_train = (X_train - np.mean(X_train, axis=(1, 2), keepdims=True)) / std_train
    
            std_test = np.std(X_test, axis=(1, 2), keepdims=True)
            std_test[std_test == 0] = 1e-6  # Reemplazar std=0 por un valor pequeño
            X_test = (X_test - np.mean(X_test, axis=(1, 2), keepdims=True)) / std_test
        else:
            raise ValueError(f"Las dimensiones de X_train son incorrectas: {X_train.shape}")
        
        # Añadir dimensión del canal para PyTorch
        X_train = X_train[..., np.newaxis]  # Dimensiones: [NWinPatID, 21, 128, 1]
        X_test = X_test[..., np.newaxis]  # Dimensiones: [NWinPatID, 21, 128, 1]
    
        # Crear datasets y DataLoaders
        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
        # Crear el modelo
        print("\nInicializando el modelo...")
        model = EEGAttentionCNN(num_channels=21).to(device)
        class_weights = torch.tensor([len(y_train) / sum(y_train == 0), len(y_train) / sum(y_train == 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Entrenar el modelo
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=30)

        # Guardar pérdidas de entrenamiento y validación
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Evaluar modelo y guardar métricas
        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = (outputs > 0.5).float()
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], zero_division=0)

        all_confusion_matrices.append(cm)
        all_classification_reports.append(report)

        # Guardar matriz de confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title(f"Matriz de Confusión - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"confusion_matrix_fold_{fold}.png")
        plt.close()

        # Guardar curva de pérdida individual
        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Curva de Pérdida - Fold {fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_curve_fold_{fold}.png")
        plt.close()

        fold += 1

    print("Proceso de validación cruzada Leave-One-Patient-Out completado.")


