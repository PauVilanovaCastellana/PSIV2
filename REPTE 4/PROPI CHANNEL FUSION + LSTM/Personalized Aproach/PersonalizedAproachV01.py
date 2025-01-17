import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F


# --- CONFIG ---
DATASET_PATH = "/export/fhome/maed/EpilepsyDataSet/"
RESULTS_DIR = "/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS PersonalizedAproach/Propi CF FW"
CHANNEL_FUSION_MODEL_PATH = "/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS PersonalizedApproach/Modelo_fold_2.pth"
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
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((1, 2))

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
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.9)

    def _initialize_fc1(self, x):
        flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 256).to(x.device)

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
def extract_features_all_windows(patient_data, channel_fusion_model):
    """
    Extrae características para todas las ventanas de todos los pacientes
    utilizando el modelo Channel Fusion.
    """
    features_data = {}
    for patient_id, patient_info in patient_data.items():
        data = patient_info["data"]  # [n_windows, num_channels, num_points]
        labels = patient_info["labels"]  # [n_windows]

        features = []
        for i in range(0, len(data), 64):  # Procesar en lotes para reducir memoria
            batch = data[i:i + 64]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(2).to(DEVICE)
            with torch.no_grad():
                batch_features = channel_fusion_model(batch_tensor, return_features=True)
            features.append(batch_features.cpu())
        features = torch.cat(features)

        features_data[patient_id] = {"data": features, "labels": torch.tensor(labels, dtype=torch.float32)}

    return features_data

def prepare_sequence_data(patient_ids, features_data, sequence_length):
    """
    Prepara datos para los folds utilizando las características extraídas.
    """
    eeg_data = []
    labels = []

    for patient_id in patient_ids:
        data = features_data[patient_id]["data"]
        label = features_data[patient_id]["labels"]

        step = sequence_length
        for i in range(0, len(label) - sequence_length + 1, step):
            eeg_sequence = data[i:i + sequence_length]
            eeg_data.append(eeg_sequence)
            sequence_labels = label[i:i + sequence_length]
            labels.append(1 if 1 in sequence_labels else 0)

    eeg_data = torch.stack(eeg_data)
    labels = torch.tensor(labels, dtype=torch.float32)
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
            n_windows = min(len(labels), 10000) # XXXXXXXXXXXX Sense Solapament 10k
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

# -------------------------- MAIN SCRIPT --------------------------
if __name__ == "__main__":
    os.makedirs(os.path.join(RESULTS_DIR, "loss_curves"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)
    print("LSTM PERSONALIZED APPROACH WITH EXTRACTED FEATURES")

    # Cargar datos por paciente
    print("Cargando datos de ventanas...")
    patient_data = load_sequential_data_by_patient(DATASET_PATH, sequence_length=10)
    patient_ids = list(patient_data.keys())

    # Cargar modelo preentrenado de Channel Fusion
    print("Cargando modelo Channel Fusion...")
    channel_fusion_model = EEGAttentionCNN(num_channels=21).to(DEVICE)
    state_dict = torch.load(CHANNEL_FUSION_MODEL_PATH, map_location=DEVICE)
    state_dict.pop("fc1.weight", None)
    state_dict.pop("fc1.bias", None)
    channel_fusion_model.load_state_dict(state_dict, strict=False)
    channel_fusion_model.eval()

    # Extraer características de todas las ventanas
    print("Extrayendo características de ventanas...")
    features_data = extract_features_all_windows(patient_data, channel_fusion_model)

    # Preparar datos combinados para StratifiedKFold
    print("Preparando datos de secuencia...")
    all_eeg_data, all_labels = prepare_sequence_data(patient_ids, features_data, sequence_length=10)

    # Configurar Stratified K-Fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Inicializar métricas
    total_cm = np.zeros((2, 2), dtype=int)
    all_metrics = []

    # Bucle a través de los folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(all_eeg_data, all_labels)):
        print(f"\nStarting Fold {fold + 1}")

        # Dividir datos
        train_dataset = Subset(EEGDataset(all_eeg_data, all_labels), train_idx)
        test_dataset = Subset(EEGDataset(all_eeg_data, all_labels), test_idx)

        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        # Inicializar modelo
        model = EpilepsyLSTM(inputmodule_params={'n_nodes': all_eeg_data.size(2)},
                             net_params={'Lstacks': 2, 'dropout': 0.5, 'hidden_size': 256},
                             outmodule_params={'n_classes': 1, 'hd': 128}).to(DEVICE)
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
            train_losses.append(train_loss / len(train_loader))

            # Validación
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    labels = labels.squeeze()
                    outputs = model(inputs).squeeze()
                    val_loss += criterion(outputs, labels).item()
            val_losses.append(val_loss / len(test_loader))
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
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                y_test_pred.extend(preds.cpu().numpy())
                y_test_true.extend(labels.cpu().numpy())

        cm = confusion_matrix(y_test_true, y_test_pred)
        total_cm += cm

        # Reportar métricas
        test_report = classification_report(y_test_true, y_test_pred, output_dict=True, zero_division=0)
        all_metrics.append(test_report)

        # Guardar matriz de confusión
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix - Fold {fold + 1}")
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.savefig(f"{RESULTS_DIR}/confusion_matrix/confusion_matrix_fold_{fold + 1}.png")
        plt.close()

    # Guardar matriz de confusión total
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title("TOTAL Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix/confusion_matrix_total.png")
    plt.close()

    # Resumen de métricas
    metrics = ["precision", "recall", "f1-score"]
    class_names = ["0.0", "1.0"]
    metrics_summary = {metric: {cls: [] for cls in class_names} for metric in metrics}

    for fold_metrics in all_metrics:
        for cls in class_names:
            for metric in metrics:
                metrics_summary[metric][cls].append(fold_metrics[cls][metric])

    # Calcular promedios y desviaciones
    data_metrics = {
        metric: {
            "means": [np.mean(metrics_summary[metric][cls]) for cls in class_names],
            "stds": [np.std(metrics_summary[metric][cls]) for cls in class_names],
        }
        for metric in metrics
    }

    # Crear gráfica de barras
    x = np.arange(len(metrics))  # Posiciones para las métricas
    width = 0.35  # Ancho de las barras

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cls in enumerate(class_names):
        means = [data_metrics[metric]["means"][i] for metric in metrics]
        stds = [data_metrics[metric]["stds"][i] for metric in metrics]
        ax.bar(x + i * width, means, width, label=f"Clase {cls}", yerr=stds, capsize=5)

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Metric Value")
    ax.set_title("Average Metrics by Class with Standard Deviation Intervals")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix/metrics_summary.png")
    plt.close()


