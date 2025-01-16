import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch.nn.init as init

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

    # Iterar por todos los archivos disponibles
    for file in os.listdir(dataset_path):
        if file.endswith(".npz"):
            patient_id = file.split("_")[0]  # ID del paciente
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

            n_samples_per_class = min(len(class_0_indices), len(class_1_indices), 2500)
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
def prepare_data(patient_data):
    eeg_data = []
    labels = []
    for patient_id in patient_data:
        eeg_data.append(patient_data[patient_id]["data"])
        labels.append(patient_data[patient_id]["labels"])
    eeg_data = np.concatenate(eeg_data, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.float32)
    return eeg_data, labels

# Modelo CNN con atención y regularización
class EEGAttentionCNN(nn.Module):
    def __init__(self, num_channels=21):
        super(EEGAttentionCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128 // 8, 128, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc1 = None
        self.fc2 = nn.Linear(256, 1) ################################# 256
        self.dropout = nn.Dropout(0.9)
        self.sigmoid = nn.Sigmoid()

    def _initialize_fc1(self, x):
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 256).to(x.device) ################################# 256

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        channel_att = self.channel_attention(x)
        x = x * channel_att

        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self._initialize_fc1(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x
        
# Función para inicializar pesos con Xavier
def initialize_weights_xavier(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Aplicar a capas Conv2d y Linear
            init.xavier_uniform_(layer.weight)  # Inicialización Xavier Uniforme
            if layer.bias is not None:  # Verificar si la capa tiene bias
                init.zeros_(layer.bias)  # Inicializar bias en 0

# Validación cruzada con 10 folds
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PERSONALIZED APPROACH")
    print("Cargando datos...")
    patient_data = load_data_by_patient(DATASET_PATH)
    X, y = prepare_data(patient_data)

    def add_noise(data, noise_level=0.05):
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    X = add_noise(X)
    std_X = np.std(X, axis=(1, 2), keepdims=True)
    std_X[std_X == 0] = 1e-6
    X = (X - np.mean(X, axis=(1, 2), keepdims=True)) / std_X
    X = X[..., np.newaxis]

    kfold = KFold(n_splits=2, shuffle=True, random_state=42)

    fold = 1
    for train_idx, test_idx in kfold.split(X):
        print(f"Fold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        # Instanciar el modelo y pesos
        model = EEGAttentionCNN(num_channels=21).to(device)
        initialize_weights_xavier(model)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)

        def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100):
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs.squeeze(), labels).item()

                scheduler.step(val_loss)
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device)

        # Calcular métricas para entrenamiento y prueba
        model.eval()
        y_test_pred, y_test_true = [], []
        y_train_pred, y_train_true = [], []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = (outputs > 0.5).float()
                y_train_pred.extend(preds.cpu().numpy())
                y_train_true.extend(labels.cpu().numpy())

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = (outputs > 0.5).float()
                y_test_pred.extend(preds.cpu().numpy())
                y_test_true.extend(labels.cpu().numpy())

        # Imprimir métricas para el fold actual
        print(f"Metrics for Fold {fold} (Train):")
        print(classification_report(y_train_true, y_train_pred, zero_division=0))
        print(f"Metrics for Fold {fold} (Test):")
        print(classification_report(y_test_true, y_test_pred, zero_division=0))
        
        model_path = f"/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS PersonalizedApproach/Modelo_fold_128_{fold}.pth"
        torch.save(model.state_dict(), model_path)

        fold += 1
