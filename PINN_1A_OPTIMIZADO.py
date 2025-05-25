"""
PINN_1A_OPTIMIZADO.py - Funciones de Utilidad Básicas Optimizadas

Características:
- Procesamiento de datos acelerado con operaciones vectorizadas
- Funciones de generación y manipulación de señales optimizadas
- Soporte para paralelización de operaciones I/O
"""

import os
import numpy as np
import torch
from scipy import signal, ndimage
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Permitir TF32 en operaciones de matrices
    torch.backends.cudnn.allow_tf32 = True  # Permitir TF32 en convolutions

print(f"Usando dispositivo: {device}")

# Función para ventana Hann optimizada
def hann_window(size):
    """
    Crea una ventana Hann para ponderar muestras (optimizada para tensor)
    
    Args:
        size: Tamaño de la ventana
        
    Returns:
        Ventana Hann como array numpy
    """
    # Cálculo vectorizado
    n = np.arange(size)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))

# Función para generar wavelet Ricker optimizada
def ricker_wavelet(freq, length, dt):
    """
    Genera un wavelet Ricker optimizado
    
    Args:
        freq: Frecuencia dominante en Hz
        length: Duración en segundos
        dt: Intervalo de muestreo en segundos
        
    Returns:
        Wavelet Ricker normalizado
    """
    # Vectorización completa
    t = np.arange(-length/2, length/2, dt)
    arg = (np.pi * freq * t) ** 2
    # Operación vectorizada
    wavelet = (1 - 2 * arg) * np.exp(-arg)
    # Normalización eficiente
    return wavelet / np.max(np.abs(wavelet))

# Función para agregar ruido a una señal (optimizada)
def add_noise(signal_data, snr_db):
    """
    Agrega ruido gaussiano a una señal con SNR específico en dB (optimizado)
    
    Args:
        signal_data: Señal original
        snr_db: Relación señal-ruido en dB
        
    Returns:
        Señal con ruido agregado
    """
    # Cálculo vectorizado de potencia
    signal_power = np.mean(signal_data ** 2)
    
    # Cálculo eficiente de potencia de ruido
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generación vectorizada de ruido
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))
    
    # Retorno directamente la suma
    return signal_data + noise

# Función para generar traza sísmica sintética optimizada
def generate_synthetic_seismic(acoustic_impedance, wavelet):
    """
    Genera traza sísmica sintética usando convolución optimizada
    
    Args:
        acoustic_impedance: Impedancia acústica
        wavelet: Wavelet sísmico
        
    Returns:
        Traza sísmica sintética
    """
    # Pre-alocar array para mejor rendimiento
    reflection_coeff = np.zeros_like(acoustic_impedance)
    
    # Cálculo vectorizado de coeficientes de reflexión
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignorar dividir por cero
        reflection_coeff[1:] = (acoustic_impedance[1:] - acoustic_impedance[:-1]) / \
                             (acoustic_impedance[1:] + acoustic_impedance[:-1])
    
    # Reemplazar posibles NaN o Inf
    reflection_coeff = np.nan_to_num(reflection_coeff)
    
    # Convolución optimizada con FFT para señales largas
    if len(reflection_coeff) > 1000:
        seismic_trace = signal.fftconvolve(reflection_coeff, wavelet, mode='same')
    else:
        # Convolución estándar para señales cortas (más rápida en este caso)
        seismic_trace = np.convolve(reflection_coeff, wavelet, mode='same')
    
    return seismic_trace

# Función de suavizado mejorada con preservación de bordes
def advanced_smooth_predictions(predicted_impedance, real_impedance=None, window_size=15, edge_preserve_factor=0.7):
    """
    Suaviza predicciones con preservación adaptativa de bordes importantes
    
    Args:
        predicted_impedance: Predicción original con posibles picos
        real_impedance: Opcional, impedancia real para calibración
        window_size: Tamaño de ventana para suavizado
        edge_preserve_factor: Factor para preservar bordes (0-1)
        
    Returns:
        Impedancia suavizada con bordes importantes preservados
    """
    # Convertir a numpy si es tensor
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    if real_impedance is not None and isinstance(real_impedance, torch.Tensor):
        real_impedance = real_impedance.cpu().numpy()
    
    # 1. Aplicar filtro de mediana para eliminar picos extremos
    smoothed = signal.medfilt(predicted_impedance, kernel_size=window_size)
    
    # 2. Calcular gradiente para detectar cambios significativos
    gradient = np.abs(np.gradient(predicted_impedance))
    
    # 3. Normalizar gradiente
    if np.max(gradient) > 0:
        norm_gradient = gradient / np.max(gradient)
    else:
        norm_gradient = gradient
    
    # 4. Crear máscara para preservar bordes importantes
    edge_mask = np.clip(norm_gradient * 2.0, 0, 1) ** 2
    
    # 5. Aplicar suavizado adaptativo
    result = np.zeros_like(predicted_impedance)
    
    # Cálculo vectorizado
    edge_weight = edge_mask * edge_preserve_factor
    result = smoothed * (1 - edge_weight) + predicted_impedance * edge_weight
    
    # 6. Suavizado final preservando tendencias
    window_length = min(window_size + (window_size % 2 == 0), len(result))
    if window_length > 3:
        if window_length % 2 == 0:  # Ventana debe ser impar
            window_length -= 1
        result = signal.savgol_filter(result, window_length=window_length, polyorder=3)
    
    return result

# Función para leer archivos LAS con paralelización
def read_las_file(file_path):
    """
    Lee un archivo LAS y extrae densidad, velocidad e impedancia acústica
    
    Args:
        file_path: Ruta al archivo LAS
        
    Returns:
        Diccionario con datos del pozo o None si hay error
    """
    try:
        import lasio
        las = lasio.read(file_path)
        
        # Extracción eficiente de datos
        depth = las.curves["DEPT"].data
        density = las.curves["DENS"].data
        velocity = las.curves["VEL"].data
        
        # Cálculo vectorizado de impedancia
        acoustic_impedance = density * velocity
        
        return {
            'depth': depth,
            'density': density,
            'velocity': velocity,
            'acoustic_impedance': acoustic_impedance,
            'well_name': os.path.basename(file_path)
        }
    except Exception as e:
        print(f"Error leyendo archivo {file_path}: {e}")
        return None

# Función para leer múltiples archivos LAS en paralelo
def read_las_files_parallel(file_paths, max_workers=None):
    """
    Lee múltiples archivos LAS en paralelo
    
    Args:
        file_paths: Lista de rutas a archivos LAS
        max_workers: Número máximo de workers (None = automático)
        
    Returns:
        Lista de datos de pozos
    """
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)
    
    well_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(read_las_file, file_paths))
        for data in results:
            if data is not None:
                well_data.append(data)
    
    return well_data

# Funciones de preprocesamiento de datos optimizadas
def normalize_data(data, scaler=None):
    """
    Normaliza datos usando StandardScaler
    
    Args:
        data: Datos a normalizar
        scaler: StandardScaler pre-existente o None para crear uno nuevo
        
    Returns:
        Datos normalizados y scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    # Asegurar shape correcto
    data_reshaped = data.reshape(-1, 1)
    
    if scaler is None:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        normalized_data = scaler.transform(data_reshaped).flatten()
    
    return normalized_data, scaler

# Funciones para análisis de datos
def analyze_well_data(well_data):
    """
    Analiza estadísticas básicas de datos de pozo
    
    Args:
        well_data: Diccionario con datos de pozo
        
    Returns:
        Diccionario con estadísticas
    """
    impedance = well_data['acoustic_impedance']
    
    # Cálculos vectorizados
    stats = {
        'min': np.min(impedance),
        'max': np.max(impedance),
        'mean': np.mean(impedance),
        'std': np.std(impedance),
        'median': np.median(impedance),
        'range': np.max(impedance) - np.min(impedance),
        'gradient_max': np.max(np.abs(np.gradient(impedance))),
        'gradient_mean': np.mean(np.abs(np.gradient(impedance))),
        'num_samples': len(impedance)
    }
    
    return stats

# Función para verificar calidad de datos
def check_data_quality(well_data):
    """
    Verifica la calidad de los datos de pozo
    
    Args:
        well_data: Diccionario con datos de pozo
        
    Returns:
        Dict con información de calidad
    """
    depth = well_data['depth']
    density = well_data['density']
    velocity = well_data['velocity']
    impedance = well_data['acoustic_impedance']
    
    # Buscar valores nulos o infinitos
    has_null_depth = np.any(np.isnan(depth)) or np.any(np.isinf(depth))
    has_null_density = np.any(np.isnan(density)) or np.any(np.isinf(density))
    has_null_velocity = np.any(np.isnan(velocity)) or np.any(np.isinf(velocity))
    has_null_impedance = np.any(np.isnan(impedance)) or np.any(np.isinf(impedance))
    
    # Verificar espaciado de muestras
    depth_spacing = np.diff(depth)
    is_uniform_spacing = np.allclose(depth_spacing, depth_spacing[0], rtol=1e-3)
    
    # Verificar valores negativos
    has_negative_density = np.any(density < 0)
    has_negative_velocity = np.any(velocity < 0)
    
    quality_info = {
        'has_null_values': has_null_depth or has_null_density or has_null_velocity or has_null_impedance,
        'is_uniform_spacing': is_uniform_spacing,
        'has_negative_values': has_negative_density or has_negative_velocity,
        'num_samples': len(depth),
        'sample_spacing': depth_spacing[0] if is_uniform_spacing else 'non-uniform'
    }
    
    return quality_info

# Función para transformación a tensor PyTorch
def to_torch_tensor(data, dtype=torch.float32, device=None):
    """
    Convierte datos a tensor de PyTorch
    
    Args:
        data: Datos a convertir (array numpy)
        dtype: Tipo de datos para tensor
        device: Dispositivo para el tensor
        
    Returns:
        Tensor PyTorch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    else:
        return torch.tensor(data, dtype=dtype, device=device)

# Función para preprocesamiento optimizado de datos
def preprocess_well_data(well_data, add_noise_level=None, wavelet=None):
    """
    Preprocesa datos de pozo con optimizaciones
    
    Args:
        well_data: Datos de pozo
        add_noise_level: Nivel de ruido a agregar (SNR en dB) o None
        wavelet: Wavelet para generar sísmica sintética o None
    
    Returns:
        Diccionario con datos preprocesados
    """
    acoustic_impedance = well_data['acoustic_impedance']
    
    # Generar sísmica sintética si se proporciona wavelet
    seismic_trace = None
    noisy_seismic = None
    
    if wavelet is not None:
        seismic_trace = generate_synthetic_seismic(acoustic_impedance, wavelet)
        
        # Agregar ruido si se especifica
        if add_noise_level is not None:
            noisy_seismic = add_noise(seismic_trace, add_noise_level)
        else:
            noisy_seismic = seismic_trace.copy()
    
    # Devolver datos preprocesados
    processed_data = {
        'acoustic_impedance': acoustic_impedance,
        'seismic_trace': seismic_trace,
        'noisy_seismic': noisy_seismic
    }
    
    return processed_data
