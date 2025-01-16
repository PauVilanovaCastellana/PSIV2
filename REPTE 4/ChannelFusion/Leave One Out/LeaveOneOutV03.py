import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch.nn.init as init

# Ruta a los datos
DATASET_PATH = "/export/fhome/maed/EpilepsyDataSet/"
RESULTS_DIR = "/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS LeaveOneOut/"

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
            # Obtener el ID del paciente desde el nombre del archivo (e.g., chb01)
            patient_id = file.split("_")[0]  # Resultado: "chb01", "chb02", etc.
            
            # Cargar el archivo .npz
            npz_path = os.path.join(dataset_path, file)
            data = np.load(npz_path, allow_pickle=True)["EEG_win"]  # Matriz [NWinPatID, 21, 128]
            
            # Cargar el archivo .parquet asociado
            file_parts = file.split("_")
            base_name = "_".join(file_parts[:2])  # Obtener "chb01_seizure"
            suffix_number = file_parts[-1].replace(".npz", "")  # Obtener "1"
            parquet_filename = f"{base_name}_metadata_{suffix_number}.parquet"
            parquet_path = os.path.join(dataset_path, parquet_filename)
            metadata = pd.read_parquet(parquet_path)
            
            # Obtener etiquetas (class) desde el .parquet
            labels = metadata["class"].values
            
            # Separar datos por clase para este paciente
            class_0_indices = np.where(labels == 0)[0]
            class_1_indices = np.where(labels == 1)[0]
            
            # Determinar el número máximo de elementos por clase para este paciente
            n_samples_per_class = min(len(class_0_indices), len(class_1_indices), 2500)

            # Seleccionar aleatoriamente n_samples_per_class por cada clase
            selected_class_0_indices = np.random.choice(class_0_indices, n_samples_per_class, replace=False)
            selected_class_1_indices = np.random.choice(class_1_indices, n_samples_per_class, replace=False)

            # Combinar y barajar los índices seleccionados
            selected_indices = np.concatenate([selected_class_0_indices, selected_class_1_indices])
            np.random.shuffle(selected_indices)
            
            # Filtrar los datos y etiquetas con los índices seleccionados
            filtered_data = data[selected_indices]
            filtered_labels = labels[selected_indices]
            
            # Agregar los datos al diccionario de este paciente
            if patient_id not in patient_data:
                patient_data[patient_id] = {"data": [], "labels": []}
            patient_data[patient_id]["data"].append(filtered_data)
            patient_data[patient_id]["labels"].append(filtered_labels)

    # Combinar los datos de cada paciente por separado
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
    eeg_data = np.concatenate(eeg_data, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.float32)
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
        self.dropout = nn.Dropout(0.7)  
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50):
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

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses
    
# Función para inicializar pesos con Xavier
def initialize_weights_xavier(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):  # Aplicar a capas Conv2d y Linear
            init.xavier_uniform_(layer.weight)  # Inicialización Xavier Uniforme
            if layer.bias is not None:  # Verificar si la capa tiene bias
                init.zeros_(layer.bias)  # Inicializar bias en 0
                

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
    df.at[row_index, 'precision_class_0'] = test_report['0.0']['precision']
    df.at[row_index, 'recall_class_0'] = test_report['0.0']['recall']
    df.at[row_index, 'precision_class_1'] = test_report['1.0']['precision']
    df.at[row_index, 'recall_class_1'] = test_report['1.0']['recall']

    # Guardar los cambios en el archivo Excel
    df.to_excel(file_path, index=False)
    print(f"Métricas actualizadas para el paciente {patient_id} en el archivo Excel.")

    

# Validación cruzada Leave-One-Patient-Out
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("LEAVE ONE OUT")
    print("lr=0.0005, weight_decay=1e-4, DropOut=07, factor=0.05 amb Xavier")

    print("Cargando datos por paciente...")
    patient_data = load_data_by_patient(DATASET_PATH)
    patient_ids = list(patient_data.keys())

    all_train_losses = []
    all_val_losses = []
    all_metrics = []
    
    # Initialize Excel file for metrics
    metrics_file = os.path.join(RESULTS_DIR, "metrics.xlsx")
    initialize_metrics_excel(patient_ids=[], file_path=metrics_file)  # Inicialización de Excel vacío
    
    # Inicializar matriz de confusión acumulada
    total_cm = np.zeros((2, 2), dtype=int)

    fold = 1
    for test_patient in patient_ids:
        print(f"Fold {fold}: Test Patient {test_patient}")

        train_patients = [p for p in patient_ids if p != test_patient]
        X_train, y_train = prepare_data(train_patients, patient_data)
        X_test, y_test = prepare_data([test_patient], patient_data)

        # Normalizar y añadir ruido a los datos
        def add_noise(data, noise_level=0.05):
            noise = np.random.normal(0, noise_level, data.shape)
            return data + noise

        X_train = add_noise(X_train)
        X_test = add_noise(X_test)

        std_train = np.std(X_train, axis=(1, 2), keepdims=True)
        std_train[std_train == 0] = 1e-6
        X_train = (X_train - np.mean(X_train, axis=(1, 2), keepdims=True)) / std_train

        std_test = np.std(X_test, axis=(1, 2), keepdims=True)
        std_test[std_test == 0] = 1e-6
        X_test = (X_test - np.mean(X_test, axis=(1, 2), keepdims=True)) / std_test

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        train_dataset = EEGDataset(X_train, y_train)
        test_dataset = EEGDataset(X_test, y_test)
        
        print(f"Fold {fold}: Distribution of classes")
        print(f"Training set: Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        print(f"Test set: Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)}")

        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        print("\nInicializando el modelo...")
        # Instanciar el modelo y pesos
        model = EEGAttentionCNN(num_channels=21).to(device)
        initialize_weights_xavier(model)

        class_weights = torch.tensor([len(y_train) / sum(y_train == 0), len(y_train) / sum(y_train == 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=3, verbose=True)

        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=50)

        # Guardar curva de pérdida
        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Curva de Pérdida - Fold {fold}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS LeaveOneOut/loss_curves/loss_curve_fold_{fold}.png")
        plt.close()

        # Calcular y guardar métricas
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
                
        print(f"Metrics for Fold {fold} (Train):")
        print(classification_report(y_train_true, y_train_pred, zero_division=0))
        print(f"Metrics for Fold {fold} (Test):")
        test_report_print = classification_report(y_test_true, y_test_pred, zero_division=0)
        print(test_report_print)
        test_report = classification_report(y_test_true, y_test_pred, output_dict=True, zero_division=0)
        
        # Update Excel with metrics
        update_metrics_excel(test_patient, test_report, metrics_file)

        # Guardar matriz de confusión
        cm = confusion_matrix(y_test_true, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS LeaveOneOut/confusion_matrix/confusion_matrix_fold_{fold}.png")
        plt.close()
        
        total_cm += cm  # Acumular en la matriz total

        all_metrics.append(test_report)
        fold += 1

    # Calcular métricas promedio por clase
    class_names = ["0.0", "1.0"]  # Clases del reporte
    metrics = ["precision", "recall", "f1-score"]
    
    # Inicializar diccionario para acumulación
    metrics_summary = {metric: {cls: [] for cls in class_names} for metric in metrics}
    
    print(all_metrics)
    
    # Recopilar métricas por fold
    for fold_metrics in all_metrics:
        print(f"Fold metrics: {fold_metrics}")  # Depuración
        print(f"Type: {type(fold_metrics)}")   # Depuración
        for cls in class_names:
            for metric in metrics:
                print(cls, fold_metrics, metric)
                metrics_summary[metric][cls].append(fold_metrics[cls][metric])
    
    # Calcular media y desviación estándar de las métricas por clase
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
    
    # Graficar barras para cada clase
    for i, cls in enumerate(class_names):
        means = [data_metrics[metric]["means"][i] for metric in metrics]
        stds = [data_metrics[metric]["stds"][i] for metric in metrics]
        ax.bar(x + i * width, means, width, label=f"Clase {cls}", yerr=stds, capsize=5)
    
    # Configurar gráfica
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Metric Value")
    ax.set_title("Average Metrics by Class with Standard Deviation Intervals")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Mostrar gráfica
    plt.tight_layout()
    plt.savefig("/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS LeaveOneOut/metrics_summary.png")
    plt.close()

    # Guardar la matriz de confusión acumulada
    plt.figure(figsize=(10, 8))
    sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(f"TOTAL Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/ChannelFusion/RESULTS LeaveOneOut/confusion_matrix_fold_TOTAL.png")
    plt.close()

