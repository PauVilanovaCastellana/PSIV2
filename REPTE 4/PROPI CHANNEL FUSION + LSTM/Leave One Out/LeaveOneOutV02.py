# --- IMPORTS ---
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

# --- CONFIG ---
DATASET_PATH = "/export/fhome/maed/EpilepsyDataSet/"
RESULTS_DIR = "/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/Propi CF FW2"
CHANNEL_FUSION_MODEL_PATH = "/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS PersonalizedApproach/Modelo_fold_128_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CLASS ---
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data.clone().detach() if isinstance(eeg_data, torch.Tensor) else torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = labels.clone().detach() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]

# --- CHANNEL FUSION MODEL ---
class EEGAttentionCNN(nn.Module):
    def __init__(self, num_channels=21):
        super(EEGAttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 2))  # Cambiado: Reduce solo en el ancho

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))  # Cambiado: Reduce solo en el ancho

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((1, 2))  # Cambiado: Reduce solo en el ancho

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
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.9)

    def _initialize_fc1(self, x):
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 128).to(x.device)

    def forward(self, x, return_features=False):
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

        features = F.relu(self.fc1(x))
        if return_features:
            return features  # Devuelve características

        x = self.dropout(features)
        x = torch.sigmoid(self.fc2(x))
        return x


# --- LSTM MODEL ---
class EpilepsyLSTM(nn.Module):
    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()
        self.lstm = nn.LSTM(input_size=inputmodule_params['n_nodes'],
                            hidden_size=net_params['hidden_size'],
                            num_layers=net_params['Lstacks'],
                            batch_first=True,
                            dropout=net_params['dropout'])
        self.fc = nn.Sequential(
            nn.Linear(net_params['hidden_size'], outmodule_params['hd']),
            nn.ReLU(),
            nn.Linear(outmodule_params['hd'], outmodule_params['n_classes'])
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Salida de la última célula LSTM
        return self.fc(out)

# --- DATA PREPARATION ---
def prepare_sequence_data_with_model(patient_ids, patient_data, channel_fusion_model, sequence_length):
    eeg_data = []  # Lista para almacenar las secuencias EEG
    labels = []    # Lista para almacenar las etiquetas correspondientes

    for patient_id in patient_ids:
        data = patient_data[patient_id]["data"]
        label = patient_data[patient_id]["labels"]

        fused_windows = []
        # Extrae características para cada ventana con el modelo
        for window in data:
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)
            features = channel_fusion_model(window_tensor, return_features=True)  # Salida: (1, feature_dim)
            fused_windows.append(features.cpu())  # Mueve a CPU si es necesario

        fused_windows = torch.stack(fused_windows)  # Tensor de forma (num_windows, 1, feature_dim)

        # Divide en secuencias de longitud fija
        step = sequence_length
        for i in range(0, len(label) - sequence_length + 1, step):
            # Extrae una secuencia de longitud fija
            eeg_sequence = torch.cat(tuple(fused_windows[i:i + sequence_length]), dim=0).unsqueeze(0)  # Cambiado a tupla
            eeg_data.append(eeg_sequence)

            # Determina la etiqueta de la secuencia (1 si hay al menos una crisis)
            sequence_labels = label[i:i + sequence_length]
            labels.append(1 if 1 in sequence_labels else 0)

    # Convierte a tensores finales
    eeg_data = torch.cat(eeg_data, dim=0)  # Combina las secuencias: (num_sequences, sequence_length, feature_dim)
    labels = torch.tensor(labels, dtype=torch.float32)  # Etiquetas correspondientes

    return eeg_data, labels
    
def load_sequential_data_by_patient(dataset_path, sequence_length=10):
    """
    Carga los datos de cada paciente y selecciona hasta X ventanas.
    """
    patient_data = {}

    # Iterar por todos los archivos disponibles
    for file in os.listdir(dataset_path):
        if file.endswith(".npz"):
            patient_id = file.split("_")[0]  # Obtener "chb01", "chb02", etc.
            npz_path = os.path.join(dataset_path, file)

            # Cargar datos y asegurarse de que sean numéricos
            data = np.load(npz_path, allow_pickle=True)["EEG_win"]
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Archivo {file} no contiene un ndarray válido.")
            data = np.array(data, dtype=np.float32)  # Convertir a float32

            # Cargar etiquetas desde el archivo .parquet
            file_parts = file.split("_")
            base_name = "_".join(file_parts[:2])
            suffix_number = file_parts[-1].replace(".npz", "")
            parquet_filename = f"{base_name}_metadata_{suffix_number}.parquet"
            parquet_path = os.path.join(dataset_path, parquet_filename)
            metadata = pd.read_parquet(parquet_path)

            labels = metadata["class"].values

            # Seleccionar hasta X ventanas y etiquetas
            n_windows = min(len(labels), 5000) # XXXXXXXXXXXX Sense Solapament 10k
            filtered_data = data[:n_windows]
            filtered_labels = labels[:n_windows]

            # Almacenar datos en el diccionario
            patient_data[patient_id] = {"data": filtered_data, "labels": filtered_labels}

    return patient_data
        
# -------------------------- METRICS TO EXCEL ---------------------
def initialize_metrics_excel(patient_ids, file_path):
    """
    Inicializa un archivo Excel con columnas para almacenar métricas.
    """
    # Crear DataFrame vacío con las columnas necesarias
    df = pd.DataFrame({
        'subject': patient_ids,
        'precision_class_0': [None] * len(patient_ids),
        'recall_class_0': [None] * len(patient_ids),
        'precision_class_1': [None] * len(patient_ids),
        'recall_class_1': [None] * len(patient_ids),
    })
    
    # Guardar DataFrame en el archivo Excel
    df.to_excel(file_path, index=False)
    print(f"Archivo Excel inicializado en: {file_path}")



def update_metrics_excel(patient_id, test_report, file_path):
    """
    Actualiza el archivo Excel con las métricas de un paciente.
    """
    # Cargar el archivo Excel
    df = pd.read_excel(file_path)

    # Verificar si el paciente ya está en el DataFrame
    if patient_id not in df['subject'].values:
        # Si no está, añadir una nueva fila
        new_row = {
            'subject': patient_id,
            'precision_class_0': None,
            'recall_class_0': None,
            'precision_class_1': None,
            'recall_class_1': None,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Actualizar las métricas para el paciente
    row_index = df[df['subject'] == patient_id].index[0]
    df.at[row_index, 'precision_class_0'] = test_report.get('0.0', {}).get('precision', None)
    df.at[row_index, 'recall_class_0'] = test_report.get('0.0', {}).get('recall', None)
    df.at[row_index, 'precision_class_1'] = test_report.get('1.0', {}).get('precision', None)
    df.at[row_index, 'recall_class_1'] = test_report.get('1.0', {}).get('recall', None)

    # Guardar los cambios en el archivo Excel
    df.to_excel(file_path, index=False)
    print(f"Métricas actualizadas para el paciente {patient_id} en el archivo Excel.")


    
# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("LSTM LEAVE ONE OUT WITH CHANNEL FUSION")
    os.makedirs(os.path.join(RESULTS_DIR, "loss_curves"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)

    metrics_file = os.path.join(RESULTS_DIR, "metrics.xlsx")
    initialize_metrics_excel(patient_ids=[], file_path=metrics_file)  # Inicialización de Excel vacío

    # Cargar datos por paciente
    patient_data = load_sequential_data_by_patient(DATASET_PATH, sequence_length=10)
    patient_ids = list(patient_data.keys())

    # Cargar el modelo preentrenado de Channel Fusion
    channel_fusion_model = EEGAttentionCNN(num_channels=21).to(DEVICE)
    state_dict = torch.load(CHANNEL_FUSION_MODEL_PATH, map_location=DEVICE)
    
    # Eliminar las claves no necesarias del state_dict
    state_dict.pop("fc1.weight", None)
    state_dict.pop("fc1.bias", None)
    
    # Cargar el state_dict modificado
    channel_fusion_model.load_state_dict(state_dict, strict=False)
    channel_fusion_model.eval()

    total_cm = np.zeros((2, 2))
    all_metrics = []

    # Crear el archivo para guardar las distribuciones de clases
    distribution_file = os.path.join(RESULTS_DIR, "class_distributions.txt")
    with open(distribution_file, "w") as f:
        f.write("Class Distributions for Each Fold\n")
        f.write("=" * 40 + "\n")

    for fold, test_patient in enumerate(patient_ids):
        print(f"\nFold {fold + 1} - Test Patient: {test_patient}")

        # Obtener pacientes de entrenamiento y prueba
        train_patients = [pid for pid in patient_ids if pid != test_patient]
        X_train, y_train = prepare_sequence_data_with_model(train_patients, patient_data, channel_fusion_model, sequence_length=10)
        X_test, y_test = prepare_sequence_data_with_model([test_patient], patient_data, channel_fusion_model, sequence_length=10)

        # Calcular la distribución de clases
        train_class_0 = sum(y_train == 0).item()
        train_class_1 = sum(y_train == 1).item()
        test_class_0 = sum(y_test == 0).item()
        test_class_1 = sum(y_test == 1).item()

        # Guardar las distribuciones en el archivo de texto
        with open(distribution_file, "a") as f:
            f.write(f"Fold {fold + 1} - Test Patient: {test_patient}\n")
            f.write(f"Train: Clase 0: {train_class_0}, Clase 1: {train_class_1}\n")
            f.write(f"Test: Clase 0: {test_class_0}, Clase 1: {test_class_1}\n")
            f.write("-" * 40 + "\n")

        # Crear loaders para entrenamiento y prueba
        train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=64, shuffle=True)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=64, shuffle=False)

        # Inicializar modelo LSTM
        model = EpilepsyLSTM(
            inputmodule_params={'n_nodes': X_train.size(2)},
            net_params={'Lstacks': 2, 'dropout': 0.5, 'hidden_size': 256},
            outmodule_params={'n_classes': 1, 'hd': 128}
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = nn.BCEWithLogitsLoss()

        # Entrenamiento
        train_losses, val_losses = [], []
        for epoch in range(30):  # Número de épocas
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                labels = labels.squeeze()

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validación
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    labels = labels.squeeze()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(test_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

        # Guardar curva de pérdida
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curve - Fold {fold + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{RESULTS_DIR}/loss_curves/loss_curve_fold_{fold + 1}.png")
        plt.close()

        # Evaluación
        y_test_pred, y_test_true = [], []
        y_train_pred, y_train_true = [], []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                y_train_pred.extend(preds.cpu().numpy())
                y_train_true.extend(labels.cpu().numpy())

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                y_test_pred.extend(preds.cpu().numpy())
                y_test_true.extend(labels.cpu().numpy())

        # Reporte de métricas
        print(f"Metrics for Fold {fold + 1} (Train):")
        print(classification_report(y_train_true, y_train_pred, zero_division=0))
        print(f"Metrics for Fold {fold + 1} (Test):")
        test_report_print = classification_report(y_test_true, y_test_pred, zero_division=0)
        print(test_report_print)
        test_report = classification_report(y_test_true, y_test_pred, output_dict=True, zero_division=0)

        # Actualizar Excel
        update_metrics_excel(test_patient, test_report, metrics_file)

        # Matriz de confusión
        cm = confusion_matrix(y_test_true, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix - Fold {fold + 1}")
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.savefig(f"{RESULTS_DIR}/confusion_matrix/confusion_matrix_fold_{fold + 1}.png")
        plt.close()

        total_cm += cm
        all_metrics.append(test_report)

    # Guardar métricas finales y matriz de confusión total
    save_final_metrics_and_plots(all_metrics, total_cm, RESULTS_DIR)
