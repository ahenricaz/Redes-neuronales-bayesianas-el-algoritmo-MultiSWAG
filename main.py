import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from copy import deepcopy
import math
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def prepare_financial_data(
    symbol: str = 'SPY',
    start_date: str = '2015-01-01',
    sequence_length: int = 30
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, pd.DatetimeIndex]:
    """
    Prepara y procesa datos financieros para el entrenamiento del modelo MultiSWAG.
    
    El proceso incluye:
    1. Descarga de datos históricos del mercado
    2. Cálculo de indicadores técnicos
    3. Preprocesamiento y normalización
    4. Creación de secuencias temporales
    5. División en conjuntos de entrenamiento/validación/test
    
    Args:
        symbol: Símbolo del activo financiero (por defecto 'SPY' para S&P 500)
        start_date: Fecha inicial para los datos históricos
        sequence_length: Longitud de las secuencias temporales a crear
        
    Returns:
        Tuple con:
        - DataLoader de entrenamiento
        - DataLoader de validación
        - DataLoader de test
        - Scaler para la variable objetivo
        - Fechas correspondientes al conjunto de test
        
    Notas:
    - Los datos se normalizan usando StandardScaler
    - La división es 70% train, 15% validación, 15% test
    - Se incluyen múltiples indicadores técnicos como features
    """
    # Descarga de datos históricos
    print(f"Downloading data for {symbol}...")
    df = yf.download(symbol, start=start_date)
    df_features = pd.DataFrame(index=df.index)
    
    print("Calculating technical indicators...")
    
    # Características básicas del mercado
    # Returns: Retornos diarios simples
    df_features['Returns'] = df['Close'].pct_change()
    # LogReturns: Retornos logarítmicos para mejor normalidad
    df_features['LogReturns'] = np.log1p(df_features['Returns'])
    # Volume_Change: Cambios en volumen como indicador de actividad
    df_features['Volume_Change'] = df['Volume'].pct_change()
    
    # Medias móviles para diferentes ventanas temporales
    for window in [10, 20, 50]:  # Corto, medio y largo plazo
        sma = df['Close'].rolling(window=window).mean()
        # SMA: Media móvil simple
        df_features[f'SMA_{window}'] = sma
        # SMA_Norm: Normalizada respecto al precio actual
        df_features[f'SMA_{window}_Norm'] = (df['Close'] / sma) - 1
    
    # Volatilidad realizada (20 días)
    df_features['RealizedVol'] = df_features['Returns'].rolling(window=20).std() * np.sqrt(252)
    
    # RSI (Relative Strength Index) - 14 días
    delta = df['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = -loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()  # Media móvil exponencial corta
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()  # Media móvil exponencial larga
    df_features['MACD'] = exp12 - exp26
    
    # Limpieza de datos
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna()
    
    # Almacenamiento de fechas para posterior uso
    dates = df_features.index
    
    # Preparación de arrays para modelo
    X = df_features.values  # Features
    y = df.loc[df_features.index, 'Close'].values  # Variable objetivo (precios)
    
    # Normalización de datos
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Creación de secuencias temporales
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_sequences.append(y_scaled[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # División de datos
    test_size = int(0.15 * len(X_sequences))
    val_size = int(0.15 * len(X_sequences))
    train_size = len(X_sequences) - test_size - val_size
    
    # Creación de DataLoaders
    try:
        # DataLoader de entrenamiento con shuffle
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_sequences[:train_size]),
                torch.FloatTensor(y_sequences[:train_size])
            ),
            batch_size=64,
            shuffle=True,
            num_workers=0  # Desactivado para debugging
        )
        
        # DataLoader de validación
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_sequences[train_size:train_size+val_size]),
                torch.FloatTensor(y_sequences[train_size:train_size+val_size])
            ),
            batch_size=64
        )
        
        # DataLoader de test
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_sequences[train_size+val_size:]),
                torch.FloatTensor(y_sequences[train_size+val_size:])
            ),
            batch_size=64
        )
    except Exception as e:
        print(f"Error creating DataLoaders: {str(e)}")
        raise
    
    # Fechas correspondientes al conjunto de test
    test_dates = dates[sequence_length:][train_size+val_size:]
    
    # Información sobre las dimensiones de los datos
    print(f"Feature dimensionality: {X.shape[1]}")
    print(f"Number of training sequences: {train_size}")
    print(f"Number of validation sequences: {val_size}")
    print(f"Number of test sequences: {test_size}")
    
    return train_loader, val_loader, test_loader, scaler_y, test_dates

class FinancialNet(nn.Module):
    """
    Arquitectura de red neuronal optimizada para predicción financiera.
    
    Características principales:
    1. LSTM bidireccional para capturar patrones temporales en ambas direcciones
    2. Múltiples capas para mayor capacidad de modelado
    3. Regularización mediante dropout y normalización
    4. Activación GELU para mejor convergencia
    
    La arquitectura consiste en:
    1. LSTM bidireccional con 2 capas
    2. Capa fully-connected con normalización
    3. Activación GELU y dropout
    4. Capa de salida lineal
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Inicializa la arquitectura de la red.
        
        Args:
            input_dim: Dimensión de entrada (número de features)
            hidden_dim: Dimensión del estado oculto de LSTM
        """
        super().__init__()
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_dim,    # Dimensión de features de entrada
            hidden_size=hidden_dim,  # Dimensión del estado oculto
            num_layers=2,            # Número de capas LSTM
            dropout=0.2,             # Dropout entre capas LSTM
            batch_first=True,        # Formato de entrada (batch, seq_len, features)
            bidirectional=True       # LSTM bidireccional
        )
        
        # Capas fully-connected con regularización
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 por ser bidireccional
            nn.LayerNorm(hidden_dim),               # Normalización de capa
            nn.GELU(),                              # Activación GELU
            nn.Dropout(0.2),                        # Dropout adicional
            nn.Linear(hidden_dim, 1)                # Capa de salida
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.
        
        Args:
            x: Tensor de entrada de forma (batch, sequence_length, features)
            
        Returns:
            Predicción de forma (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])  # Usa solo el último estado oculto

def set_seed(seed: int = 42):
    """
    Establece semillas aleatorias para reproducibilidad.
    
    Fija las semillas para:
    1. Python random
    2. NumPy
    3. PyTorch (CPU)
    4. PyTorch (GPU si está disponible)
    5. CuDNN
    
    Args:
        seed: Valor de la semilla a utilizar
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SWAG(nn.Module):
    """
    Implementación de Stochastic Weight Averaging - Gaussian (SWAG)
    
    SWAG es una técnica que extiende SWA (Stochastic Weight Averaging) para capturar
    la incertidumbre en el aprendizaje profundo. El proceso tiene dos componentes principales:
    
    1. SWA: Promedia los pesos (w_i) durante el descenso por gradiente para obtener
       un peso óptimo (w_SWA) que generaliza mejor que un único modelo.
    
    2. Modelado Gaussiano: Usa w_SWA como media de una distribución normal y los
       pesos recolectados para estimar la matriz de covarianza. Esta distribución
       permite muestrear diferentes versiones del modelo para estimar la incertidumbre.
    """
    def __init__(
        self,
        base_model: nn.Module,
        max_rank: int = 20,
        var_clamp: float = 1e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicialización del modelo SWAG.
        
        Args:
            base_model: Modelo base a partir del cual se construirá SWAG
            max_rank: Rango máximo para la aproximación de bajo rango de la covarianza
            var_clamp: Valor mínimo para la varianza para evitar inestabilidad numérica
            device: Dispositivo donde se ejecutará el modelo
            
        La inicialización configura:
        1. AveragedModel de PyTorch para implementar SWA
        2. Buffers para almacenar estadísticas necesarias para SWAG:
           - mean: Vector de medias (w_SWA)
           - sq_mean: Momento de segundo orden para la diagonal de covarianza
           - deviations: Matriz para aproximación de bajo rango de covarianza
        """
        super().__init__()
        self.base_model = AveragedModel(base_model)
        self.device = device
        self.n_params = sum(p.numel() for p in base_model.parameters())
        
        self.register_buffer('mean', torch.zeros(self.n_params))
        self.register_buffer('sq_mean', torch.zeros(self.n_params))
        self.register_buffer('deviations', torch.zeros(max_rank, self.n_params))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))
        
        self.max_rank = max_rank
        self.var_clamp = var_clamp
        
    def collect_model(self, model: nn.Module) -> None:
        """
        Actualiza las estadísticas de SWAG usando un nuevo modelo del proceso de optimización.
        
        Este método implementa dos aspectos clave:
        1. Actualización de w_SWA usando el promedio móvil de los pesos
        2. Actualización de las estadísticas para la matriz de covarianza:
           - Actualiza momentos para el término diagonal
           - Actualiza matriz de desviaciones para el término de bajo rango
           
        Args:
            model: Nuevo modelo cuyas estadísticas serán incorporadas
        """
        self.base_model.update_parameters(model)
        
        w = torch.cat([p.data.view(-1) for p in self.base_model.module.parameters()])
        
        if self.n_models == 0:
            self.mean.copy_(w)
            self.sq_mean.copy_(w ** 2)
        else:
            n = self.n_models.item()
            self.mean.mul_(n / (n + 1.0)).add_(w / (n + 1.0))
            self.sq_mean.mul_(n / (n + 1.0)).add_((w ** 2) / (n + 1.0))
        
        if self.n_models < self.max_rank:
            self.deviations[self.n_models].copy_(w - self.mean)
        else:
            self.deviations = torch.roll(self.deviations, -1, dims=0)
            self.deviations[-1].copy_(w - self.mean)
        
        self.n_models += 1
        
    def sample(self, scale: float = 1.0, diag_noise: bool = True) -> nn.Module:
        """
        Muestrea un modelo de la distribución posterior aproximada por SWAG.
        
        El muestreo sigue estos pasos:
        1. Comienza con w_SWA como punto base
        2. Añade ruido gaussiano de la diagonal de la covarianza
        3. Añade ruido gaussiano del componente de bajo rango
        
        Args:
            scale: Factor de escala para el ruido (controla la magnitud de la incertidumbre)
            diag_noise: Si se debe incluir el ruido diagonal
            
        Returns:
            Nuevo modelo con pesos muestreados de la distribución SWAG
        """
        sampled_params = self.mean.clone()
        
        if self.n_models > 0:
            if diag_noise:
                var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
                sampled_params.add_(
                    scale * torch.randn_like(sampled_params) * torch.sqrt(var)
                )
            
            if self.n_models > 1:
                rank = min(self.n_models.item(), self.max_rank)
                z1 = torch.randn(rank, device=self.device)
                z2 = torch.randn(rank, device=self.device)
                
                sampled_params.add_(
                    scale/math.sqrt(2.0) * self.deviations[:rank].t().mv(z1)
                ).add_(
                    scale/math.sqrt(2.0) * self.deviations[:rank].t().mv(z2)
                )
        
        new_model = deepcopy(self.base_model.module)
        offset = 0
        for p in new_model.parameters():
            numel = p.numel()
            p.data = sampled_params[offset:offset + numel].view_as(p)
            offset += numel
        
        return new_model


class MultiSWAG:
    """
    Implementación de Multiple SWAG que ejecuta varias instancias de SWAG
    para obtener estimaciones más robustas de la incertidumbre.
    """
    def __init__(
        self,
        base_model: nn.Module,
        num_models: int = 3,
        max_rank: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Inicialización de MultiSWAG.
        
        Crea múltiples instancias independientes de SWAG, cada una con:
        1. Una inicialización diferente de pesos
        2. Su propia trayectoria de entrenamiento
        3. Su propia distribución gaussiana resultante
        
        Args:
            base_model: Modelo base para crear las instancias SWAG
            num_models: Número de instancias SWAG independientes
            max_rank: Rango máximo para la aproximación de covarianza
            device: Dispositivo donde se ejecutará el modelo
        """
        set_seed(42)
        self.swag_models = []
        self.num_models = num_models
        self.device = device
        
        for _ in range(num_models):
            model = deepcopy(base_model)
            for param in model.parameters():
                if len(param.shape) > 1:
                    nn.init.orthogonal_(param)
            swag = SWAG(model, max_rank=max_rank, device=device)
            swag.to(device)
            self.swag_models.append(swag)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 150,
        swa_start: int = 100,
        lr_init: float = 2e-4
    ) -> List[Dict]:
        """
        Entrena todas las instancias SWAG independientemente.
        
        El proceso para cada instancia es:
        1. Entrenamiento SGD normal hasta swa_start
        2. Después de swa_start:
           - Actualiza los pesos SWA
           - Recolecta estadísticas para la covarianza SWAG
           
        Args:
            train_loader: Datos de entrenamiento
            val_loader: Datos de validación (opcional)
            epochs: Número total de épocas
            swa_start: Época donde comienza la fase SWA/SWAG
            lr_init: Tasa de aprendizaje inicial
            
        Returns:
            Lista con resultados de entrenamiento para cada modelo
        """
        results = []
        
        for i, swag_model in enumerate(self.swag_models):
            print(f"\nTraining SWAG instance {i+1}/{self.num_models}")
            
            optimizer = torch.optim.AdamW(
                swag_model.base_model.module.parameters(),
                lr=lr_init,
                weight_decay=0.01
            )
            
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            
            criterion = nn.HuberLoss(delta=1.0)
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                swag_model.base_model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = swag_model.base_model(batch_X)
                    loss = criterion(output, batch_y.view(-1, 1))
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(
                        swag_model.base_model.parameters(),
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                
                # Fase SWA/SWAG
                if epoch >= swa_start:
                    swag_model.collect_model(swag_model.base_model.module)
                else:
                    scheduler.step()
                
                if val_loader is not None:
                    val_loss = self._validate(
                        swag_model.base_model,
                        val_loader,
                        criterion
                    )
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience and epoch >= swa_start:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                    if val_loader is not None:
                        print(f"Validation Loss: {val_loss:.4f}")
            
            results.append({
                'model_id': i,
                'final_loss': avg_loss,
                'best_val_loss': best_loss if val_loader else None
            })
        
        return results

    def predict(
        self,
        X: torch.Tensor,
        num_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera predicciones combinando todas las instancias SWAG.
        
        El proceso es:
        1. Para cada modelo SWAG:
           - Genera num_samples muestras de su distribución
           - Obtiene predicciones para cada muestra
        2. Combina todas las predicciones para obtener:
           - Media (predicción final)
           - Desviación estándar (incertidumbre estimada)
           
        Args:
            X: Datos de entrada
            num_samples: Número de muestras por modelo SWAG
            
        Returns:
            Tupla con (predicciones, incertidumbre)
        """
        X = X.to(self.device)
        all_predictions = []
        
        for swag_model in self.swag_models:
            model_predictions = []
            
            for _ in range(num_samples):
                sampled_model = swag_model.sample()
                sampled_model.eval()
                
                with torch.no_grad():
                    pred = sampled_model(X)
                model_predictions.append(pred)
            
            model_predictions = torch.stack(model_predictions)
            all_predictions.append(model_predictions)
        
        all_predictions = torch.stack(all_predictions)
        mean_pred = all_predictions.mean(dim=(0, 1))
        uncertainty = all_predictions.std(dim=(0, 1))
        
        return mean_pred, uncertainty

    
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    scaler: StandardScaler
) -> Dict[str, float]:
    """
    Evalúa las predicciones y calcula métricas de rendimiento del modelo MultiSWAG.
    
    El proceso de evaluación incluye:
    1. Transformación inversa de las predicciones a la escala original
    2. Cálculo de métricas de error (RMSE, MAE, R²)
    3. Cálculo de probabilidades de cobertura para intervalos de confianza
    4. Evaluación de la incertidumbre estimada
    
    Args:
        y_true: Valores reales (ground truth)
        y_pred: Valores predichos por el modelo
        uncertainty: Incertidumbres estimadas por MultiSWAG
        scaler: Transformador usado para escalar la variable objetivo
        
    Returns:
        Tupla con:
        - Dictionary con métricas de evaluación
        - Tupla (y_true, y_pred, uncertainty) en escala original
        
    Notas:
    - Coverage_50: Usa ±0.674σ para intervalo de confianza del 50%
    - Coverage_95: Usa ±1.96σ para intervalo de confianza del 95%
    - pred_range: Rango completo de las predicciones (max - min)
    """
    # Transformación a escala original para una evaluación realista
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # La incertidumbre se escala multiplicando por la desviación estándar original
    uncertainty_orig = uncertainty.flatten() * scaler.scale_[0]
    
    # Cálculo de métricas comprehensivas
    metrics = {
        # Métricas estándar de error de predicción
        'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        'mae': mean_absolute_error(y_true_orig, y_pred_orig),
        'r2': r2_score(y_true_orig, y_pred_orig),
        
        # Probabilidades de cobertura para evaluar la calidad de la incertidumbre
        # Intervalo de confianza del 50% (±0.674σ)
        'coverage_50': np.mean(
            (y_true_orig >= y_pred_orig - 0.674 * uncertainty_orig) & 
            (y_true_orig <= y_pred_orig + 0.674 * uncertainty_orig)
        ),
        # Intervalo de confianza del 95% (±1.96σ)
        'coverage_95': np.mean(
            (y_true_orig >= y_pred_orig - 1.96 * uncertainty_orig) & 
            (y_true_orig <= y_pred_orig + 1.96 * uncertainty_orig)
        ),
        
        # Métricas adicionales de incertidumbre
        'mean_uncertainty': np.mean(uncertainty_orig),  # Incertidumbre promedio
        'pred_range': np.ptp(y_pred_orig)  # Rango total de predicciones
    }
    
    # Retorna tanto las métricas como los valores transformados para visualización
    return metrics, (y_true_orig, y_pred_orig, uncertainty_orig)

def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    metrics: Dict[str, float],
    save_path: str
) -> None:
    """
    Genera una visualización completa de las predicciones con bandas de incertidumbre.
    
    La visualización incluye:
    1. Serie temporal de valores reales (azul)
    2. Predicciones del modelo MultiSWAG (rojo)
    3. Bandas de incertidumbre (±2σ para intervalo de confianza del 95%)
    4. Métricas principales en el título (RMSE y Coverage)
    
    Args:
        dates: Índice temporal para el eje X
        y_true: Valores reales del S&P 500
        y_pred: Predicciones del modelo
        uncertainty: Incertidumbre estimada para cada predicción
        metrics: Diccionario con métricas de evaluación
        save_path: Ruta donde guardar la gráfica generada
        
    Notas:
    - Las bandas de incertidumbre usan alpha=0.2 para mejor visualización
    - Se incluye una cuadrícula para facilitar la lectura
    - Las fechas se rotan 45° para mejor legibilidad
    """
    # Configuración inicial de la figura
    plt.figure(figsize=(15, 8))
    
    # Conversión de fechas para compatibilidad con matplotlib
    dates = dates.to_numpy()
    
    # Visualización de valores reales y predicciones
    plt.plot(dates, y_true, label='Real', color='blue', alpha=0.6)
    plt.plot(dates, y_pred, label='Predicción MultiSWAG', color='red', alpha=0.6)
    
    # Bandas de incertidumbre (intervalo de confianza del 95%)
    plt.fill_between(
        dates,
        y_pred - 2 * uncertainty,  # Límite inferior
        y_pred + 2 * uncertainty,  # Límite superior
        alpha=0.2,
        color='red',
        label='Intervalo de Confianza 95%'
    )
    
    # Configuración del título con métricas principales
    plt.title(
        f'Predicciones MultiSWAG para S&P 500\n' + 
        f'RMSE: {metrics["rmse"]:.2f}, ' + 
        f'Coverage 95%: {metrics["coverage_95"]:.1%}'
    )
    
    # Mejoras de visualización
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardado de la figura
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento y evaluación.
    
    El proceso completo incluye:
    1. Configuración inicial:
       - Creación de directorio para resultados
       - Configuración del dispositivo (CPU/GPU)
       - Establecimiento de semilla aleatoria
       
    2. Preparación de datos:
       - Carga y procesamiento de datos del S&P 500
       - Creación de dataloaders para train/val/test
       
    3. Inicialización del modelo:
       - Creación del modelo base (FinancialNet)
       - Configuración de MultiSWAG con 4 instancias
       
    4. Entrenamiento:
       - 300 épocas totales
       - Fase SWA/SWAG comienza en época 200
       - Learning rate inicial de 2e-4
       
    5. Evaluación:
       - Generación de predicciones en conjunto de test
       - Cálculo de métricas de rendimiento
       - Visualización de resultados
       
    6. Almacenamiento de resultados:
       - Gráficas de predicciones
       - Métricas en formato CSV
       - Timestamps únicos para cada ejecución
    """
    # 1. Configuración inicial
    print("Setting up...")
    Path("results").mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_seed(42)  # Reproducibilidad
    
    # 2. Preparación de datos
    print("\nPreparing data...")
    train_loader, val_loader, test_loader, scaler_y, test_dates = prepare_financial_data(
        symbol='SPY',
        start_date='2015-01-01'
    )
    
    # Obtención de dimensionalidad de entrada
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"Input dimension from data: {input_dim}")
    
    # 3. Inicialización del modelo
    print("\nInitializing model...")
    base_model = FinancialNet(
        input_dim=input_dim,
        hidden_dim=256
    ).to(device)
    
    multiswag = MultiSWAG(
        base_model=base_model,
        num_models=4,  # 4 instancias independientes
        max_rank=30,   # Rango para aproximación de covarianza
        device=device
    )
    
    # 4. Configuración del entrenamiento
    training_config = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'epochs': 300,        # Épocas totales
        'swa_start': 200,    # Inicio de fase SWA/SWAG
        'lr_init': 2e-4      # Learning rate inicial
    }
    
    # Entrenamiento del modelo
    print("\nTraining model...")
    training_results = multiswag.train(**training_config)
    
    # 5. Evaluación del modelo
    print("\nEvaluating model...")
    all_predictions = []
    all_uncertainties = []
    true_values = []
    
    # Generación de predicciones en conjunto de test
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions, uncertainty = multiswag.predict(batch_X)
            all_predictions.extend(predictions.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
            true_values.extend(batch_y.numpy())
    
    # Conversión a arrays numpy
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    true_values = np.array(true_values)
    
    # Cálculo de métricas
    metrics, (y_true_orig, y_pred_orig, uncertainty_orig) = evaluate_predictions(
        true_values,
        all_predictions,
        all_uncertainties,
        scaler_y
    )
    
    # Impresión de resultados
    print("\nFinal Results:")
    for metric_name, value in metrics.items():
        if metric_name.startswith('coverage'):
            print(f"{metric_name}: {value:.1%}")
        else:
            print(f"{metric_name}: {value:.2f}")
    
    # 6. Almacenamiento de resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generación de gráficas
    plot_predictions(
        test_dates,
        y_true_orig,
        y_pred_orig,
        uncertainty_orig,
        metrics,
        f'results/predictions_{timestamp}.png'
    )
    
    # Guardado de métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'results/metrics_{timestamp}.csv')
    
    print("\nResults saved in results/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()