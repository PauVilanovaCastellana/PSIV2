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

# Main directory for the dataset
DATASET_PATH = "/export/fhome/maed/EpilepsyDataSet/"
RESULTS_DIR = "/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/"

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------- DATASET ---------------------------------
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data.clone().detach() if isinstance(eeg_data, torch.Tensor) else torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = labels.clone().detach() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


# ---------------------- DATA PREPARATION FUNCTIONS ----------------------
def fuse_channels(data):
    """
    Aplana los canales de las ventanas EEG.
    """
    data = np.array(data, dtype=np.float32)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor.view(data_tensor.size(0), -1)

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


def prepare_sequence_data(patient_ids, patient_data, sequence_length):
    """
    Prepara los datos en forma de secuencias, con etiquetas 1 si alguna ventana en la secuencia tiene clase 1.
    """
    eeg_data = []
    labels = []

    for patient_id in patient_ids:
        data = patient_data[patient_id]["data"]  # [n_windows, num_channels, num_points]
        label = patient_data[patient_id]["labels"]  # [n_windows]

        '''
        # --------- AMB SOLAPAMENT --------
        # Generar hasta Y secuencias por paciente
        max_sequences = 1000  # YYYYYYYYYY 5000
        total_sequences = min((len(label) - sequence_length + 1), max_sequences)

        for i in range(total_sequences):
            # Extraer una secuencia de longitud 'sequence_length'
            eeg_sequence = data[i:i + sequence_length]  # [sequence_length, num_channels, num_points]
            flattened_sequence = fuse_channels(eeg_sequence)  # [sequence_length, num_channels * num_points]
            eeg_data.append(flattened_sequence)

            # Etiquetar la secuencia como 1 si alguna ventana tiene etiqueta 1
            sequence_labels = label[i:i + sequence_length]
            if 1 in sequence_labels:
                labels.append(1)
            else:
                labels.append(0)
        '''
        # --------- SENSE SOLAPAMENT --------
        step = sequence_length  # Evita solapamiento
        for i in range(0, len(label) - sequence_length + 1, step):
            eeg_sequence = data[i:i + sequence_length]
            flattened_sequence = fuse_channels(eeg_sequence)
            eeg_data.append(flattened_sequence)
        
            # Etiquetar la secuencia
            sequence_labels = label[i:i + sequence_length]
            if 1 in sequence_labels:
                labels.append(1)
            else:
                labels.append(0)

    # Convertir a tensores
    eeg_data = torch.stack(eeg_data)  # [num_samples, sequence_length, num_features]
    labels = torch.tensor(labels, dtype=torch.float32)  # [num_samples]
    return eeg_data, labels

# ------------------------------- LSTM MODEL -------------------------------
class EpilepsyLSTM(nn.Module):
    def __init__(self, inputmodule_params, net_params, outmodule_params):
        super().__init__()
        print('Running class:', self.__class__.__name__)
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
        out = out[:, -1, :]
        return self.fc(out)
        
        
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

    


# -------------------------- MAIN SCRIPT --------------------------
if __name__ == "__main__":
    print("LSTM LEAVE ONE OUT")
    os.makedirs(os.path.join(RESULTS_DIR, "loss_curves"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrix"), exist_ok=True)

    # Initialize Excel file for metrics
    metrics_file = os.path.join(RESULTS_DIR, "metrics.xlsx")
    initialize_metrics_excel(patient_ids=[], file_path=metrics_file)  # Inicialización de Excel vacío

    # Load data
    patient_data = load_sequential_data_by_patient(DATASET_PATH, sequence_length=10)
    patient_ids = list(patient_data.keys())

    # Initialize variables
    total_cm = np.zeros((2, 2))
    all_metrics = []

    for fold, test_patient in enumerate(patient_ids):
        print(f"\nStarting Fold {fold + 1} with Test Patient: {test_patient}")

        # Split train and test
        train_patients = [pid for pid in patient_ids if pid != test_patient]
        X_train, y_train = prepare_sequence_data(train_patients, patient_data, sequence_length=10)
        X_test, y_test = prepare_sequence_data([test_patient], patient_data, sequence_length=10)

        # DataLoaders
        train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=1024, shuffle=True)
        test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=1024, shuffle=False)

        # Initialize model
        model = EpilepsyLSTM(inputmodule_params={'n_nodes': X_train.size(2)},
                             net_params={'Lstacks': 2, 'dropout': 0.5, 'hidden_size': 256},
                             outmodule_params={'n_classes': 1, 'hd': 128}).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        train_losses, val_losses = [], []
        for epoch in range(30):  # Número de épocas
            # Entrenamiento
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Ajustar dimensiones de labels si es necesario
                labels = labels.squeeze()  # labels: [batch_size]
        
                # Forward y cálculo de pérdida
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()  # outputs: [batch_size]
                loss = criterion(outputs, labels)
                
                # Backward y optimización
                loss.backward()
                optimizer.step()
        
                # Acumular pérdida
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
        
            # Validación después de cada epoch
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

        # Save loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curve - Fold {fold + 1}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/loss_curves/loss_curve_fold_{fold + 1}.png")
        plt.close()

        # Metrics
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
                if preds.dim() == 0:  # Si es un escalar
                  preds = preds.unsqueeze(0)  # Convertir a tensor de 1D
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
        plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/confusion_matrix/confusion_matrix_fold_{fold}.png")
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
    plt.savefig("/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/confusion_matrix/metrics_summary.png")
    plt.close()
    
    # Convertir la matriz de confusión a enteros
    total_cm = np.array(total_cm).astype(int)
    
    # Total confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        total_cm,
        annot=True,
        fmt="d",  # Formato de enteros
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"]
    )
    plt.title("TOTAL Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.savefig(f"/export/fhome/maed01/myvirtualenv/REPTE4/LSTM/RESULTS LeaveOneOut/confusion_matrix/confusion_matrix_fold_TOTAL.png")
    plt.close()