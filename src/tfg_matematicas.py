import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import torch.optim as optim
from scipy import stats
from typing import Tuple, List, Dict
import time
import json
from pathlib import Path





def get_stock_data(symbol: str = 'GOOGL', start_date: str = '2020-01-01', end_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Descarga y prepara datos de acciones usando yfinance.
    
    Args:
        symbol: Símbolo de la acción
        start_date: Fecha de inicio en formato 'YYYY-MM-DD'
        end_date: Fecha final en formato 'YYYY-MM-DD'
    
    Returns:
        DataFrame con datos de la acción y características calculadas
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Calcular características técnicas
    df['Returns'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calcular indicadores adicionales
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    
    # Eliminar valores NaN
    df = df.dropna()
    
    return df

def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calcula el Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series) -> pd.Series:
    """Calcula el Moving Average Convergence Divergence (MACD)"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def prepare_data(df: pd.DataFrame, sequence_length: int = 5) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]:
    """
    Prepara los datos para el entrenamiento.
    
    Args:
        df: DataFrame con los datos de la acción
        sequence_length: Longitud de la secuencia para la ventana temporal
    
    Returns:
        Tuple con (X_tensors, y_tensors, scaler_X, scaler_y)
    """
    # Seleccionar características
    features = ['Returns', 'SMA_5', 'SMA_20', 'Volatility', 'RSI', 'MACD']
    X = df[features].values
    y = df['Close'].values
    
    # Normalizar datos
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Crear secuencias
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    # Convertir a tensores de PyTorch
    X_tensors = torch.FloatTensor(np.array(X_sequences))
    y_tensors = torch.FloatTensor(np.array(y_sequences))
    
    return X_tensors, y_tensors, scaler_X, scaler_y




class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics = {
            'training_time': [],
            'inference_time': [],
            'mse': [],
            'mae': [],
            'r2': [],
            'uncertainty_metrics': {
                'calibration_score': [],
                'sharpness': [],
                'coverage_probabilities': []
            },
            'model_params': [],
            'convergence_metrics': []
        }
    
    def _convert_to_serializable(self, obj):
        """Convierte objetos numpy a tipos Python nativos"""
        import numpy as np
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj
        
    def add_metric(self, metric_name: str, value):
        if metric_name == 'uncertainty_metrics':
            # Convertir métricas de incertidumbre
            value = self._convert_to_serializable(value)
            if 'calibration_score' in value:
                self.metrics['uncertainty_metrics']['calibration_score'].append(
                    value['calibration_score'])
            if 'sharpness' in value:
                self.metrics['uncertainty_metrics']['sharpness'].append(
                    value['sharpness'])
            if 'coverage_probabilities' in value:
                self.metrics['uncertainty_metrics']['coverage_probabilities'].append(
                    value['coverage_probabilities'])
        elif metric_name in self.metrics:
            # Convertir valor simple
            self.metrics[metric_name].append(
                self._convert_to_serializable(value))
            
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{self.experiment_name}_{timestamp}.json"
        # Convertir todo el diccionario antes de guardar
        serializable_metrics = self._convert_to_serializable(self.metrics)
        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

class StockPredictor(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.1):
        super(StockPredictor, self).__init__()
        # Flatten the input first
        self.flatten = nn.Flatten()
        
        # Ajustar las dimensiones de entrada considerando la secuencia
        self.sequence_length = 5  # Este valor debe coincidir con el sequence_length en prepare_data
        input_size = input_dim * self.sequence_length
        
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # BatchNorm después de flatten
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.flatten(x)  # Aplanar la entrada
        x = self.dropout(self.relu(self.batch_norm1(self.layer1(x))))
        x = self.dropout(self.relu(self.batch_norm2(self.layer2(x))))
        x = self.dropout(self.relu(self.layer3(x)))
        return self.output(x)

class MultiSWAG:
    def __init__(self, base_model: nn.Module, num_models: int = 5, learning_rate: float = 0.001):
        self.models = [base_model]
        self.num_models = num_models
        self.learning_rate = learning_rate
        self.training_losses = []
        
        # Obtener las dimensiones correctas del modelo base
        input_dim = base_model.layer1.in_features // base_model.sequence_length
        
        for _ in range(num_models - 1):
            # Crear nuevo modelo con las mismas dimensiones
            new_model = StockPredictor(input_dim)
            # Ahora los state_dict deberían coincidir
            new_model.load_state_dict(base_model.state_dict())
            self.models.append(new_model)
            
    def train_models(self, train_loader, val_loader, epochs: int = 100, 
                    early_stopping_patience: int = 10) -> List[float]:
        training_metrics = []
        
        for i, model in enumerate(self.models):
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            model_losses = []
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validación
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_output = model(val_X)
                        val_loss += criterion(val_output, val_y).item()
                
                scheduler.step(val_loss)
                model_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
            
            training_metrics.append({
                'model_id': i,
                'final_loss': best_val_loss,
                'epochs_trained': len(model_losses),
                'loss_history': model_losses
            })
        
        return training_metrics

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred)
        
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        uncertainty = torch.std(torch.stack(predictions), dim=0)
        
        return ensemble_pred, uncertainty

def calculate_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                uncertainty: np.ndarray) -> Dict[str, float]:
    z_scores = (y_true - y_pred) / uncertainty
    calibration_score = np.abs(1 - np.std(z_scores))
    
    # Intervalos de confianza empíricos
    confidence_levels = [0.5, 0.9, 0.95]
    coverage_probs = {}
    for level in confidence_levels:
        interval = stats.norm.interval(level)
        coverage = np.mean((z_scores >= interval[0]) & (z_scores <= interval[1]))
        coverage_probs[f'{level*100}%_coverage'] = coverage
    
    # Sharpness (promedio de incertidumbre)
    sharpness = np.mean(uncertainty)
    
    return {
        'calibration_score': calibration_score,
        'sharpness': sharpness,
        'coverage_probabilities': coverage_probs
    }

def evaluate_models(symbols: List[str] = ['GOOGL', 'MSFT', 'AMZN'], 
                   experiment_tracker: ExperimentTracker = None) -> Dict:
    results = {}
    
    for symbol in symbols:
        # Obtener y preparar datos
        df = get_stock_data(symbol)
        X_tensors, y_tensors, scaler_X, scaler_y = prepare_data(df)
        
        # División train/validation/test
        train_size = int(0.7 * len(X_tensors))
        val_size = int(0.15 * len(X_tensors))
        
        X_train = X_tensors[:train_size]
        X_val = X_tensors[train_size:train_size+val_size]
        X_test = X_tensors[train_size+val_size:]
        
        y_train = y_tensors[:train_size]
        y_val = y_tensors[train_size:train_size+val_size]
        y_test = y_tensors[train_size+val_size:]
        
        # Crear dataloaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        
        # Inicializar y entrenar modelo
        input_dim = X_train.shape[2]
        base_model = StockPredictor(input_dim)
        multiswag = MultiSWAG(base_model)
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        training_metrics = multiswag.train_models(train_loader, val_loader)
        training_time = time.time() - start_time
        
        # Medir tiempo de inferencia
        start_time = time.time()
        multiswag_pred, uncertainty = multiswag.predict(X_test)
        inference_time = time.time() - start_time
        
        # Convertir predicciones a escala original
        y_test_orig = scaler_y.inverse_transform(y_test.numpy())
        multiswag_pred_orig = scaler_y.inverse_transform(multiswag_pred.numpy())
        uncertainty_orig = uncertainty.numpy() * scaler_y.scale_
        
        # Calcular métricas
        mse = mean_squared_error(y_test_orig, multiswag_pred_orig)
        mae = mean_absolute_error(y_test_orig, multiswag_pred_orig)
        r2 = r2_score(y_test_orig, multiswag_pred_orig)
        
        # Calcular métricas de calibración
        calibration_metrics = calculate_calibration_metrics(
            y_test_orig, multiswag_pred_orig, uncertainty_orig)
        
        # Guardar resultados
        results[symbol] = {
            'performance_metrics': {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'training_time': training_time,
                'inference_time': inference_time
            },
            'uncertainty_metrics': calibration_metrics,
            'training_metrics': training_metrics
        }
        
        if experiment_tracker:
            experiment_tracker.add_metric('mse', mse)
            experiment_tracker.add_metric('mae', mae)
            experiment_tracker.add_metric('r2', r2)
            experiment_tracker.add_metric('training_time', training_time)
            experiment_tracker.add_metric('inference_time', inference_time)
            experiment_tracker.add_metric('uncertainty_metrics', calibration_metrics)
        
        # Visualización
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Predicciones vs Valores reales
        plt.subplot(2, 1, 1)
        plt.plot(y_test_orig, label='Real', alpha=0.6)
        plt.plot(multiswag_pred_orig, label='MultiSWAG', alpha=0.6)
        plt.fill_between(range(len(multiswag_pred_orig)),
                        multiswag_pred_orig.flatten() - uncertainty_orig.flatten(),
                        multiswag_pred_orig.flatten() + uncertainty_orig.flatten(),
                        alpha=0.2)
        plt.title(f'Predicciones MultiSWAG para {symbol}')
        plt.legend()
        
        # Subplot 2: Error vs Incertidumbre
        plt.subplot(2, 1, 2)
        error = np.abs(y_test_orig - multiswag_pred_orig)
        plt.scatter(uncertainty_orig, error, alpha=0.5)
        plt.xlabel('Incertidumbre Estimada')
        plt.ylabel('Error Absoluto')
        plt.title('Relación entre Incertidumbre y Error')
        
        plt.tight_layout()
        plt.savefig(f'multiswag_analysis_{symbol}.png')
        plt.close()
        
    return results

if __name__ == "__main__":
    # Crear directorio para resultados
    Path("results").mkdir(exist_ok=True)
    
    # Inicializar tracker de experimentos
    experiment_tracker = ExperimentTracker("multiswag_analysis")
    
    # Ejecutar evaluación
    results = evaluate_models(experiment_tracker=experiment_tracker)
    
    # Guardar resultados
    experiment_tracker.save_results()
    
    # Imprimir resumen de resultados
    print("\nResumen de Resultados:")
    for symbol, metrics in results.items():
        print(f"\nResultados para {symbol}:")
        print(f"MSE: {metrics['performance_metrics']['mse']:.4f}")
        print(f"MAE: {metrics['performance_metrics']['mae']:.4f}")
        print(f"R²: {metrics['performance_metrics']['r2']:.4f}")
        print(f"Tiempo de entrenamiento: {metrics['performance_metrics']['training_time']:.2f}s")
        print(f"Tiempo de inferencia: {metrics['performance_metrics']['inference_time']:.2f}s")
        print("\nMétricas de calibración:")
        print(f"Score de calibración: {metrics['uncertainty_metrics']['calibration_score']:.4f}")
        print(f"Sharpness: {metrics['uncertainty_metrics']['sharpness']:.4f}")
        for level, prob in metrics['uncertainty_metrics']['coverage_probabilities'].items():
            print(f"Cobertura {level}: {prob:.4f}")