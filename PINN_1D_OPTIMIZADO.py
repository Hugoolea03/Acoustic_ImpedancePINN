"""
PINN_1D_OPTIMIZADO.py - Funciones de Predicción y Ensemble Optimizadas
PARTE 1: Funciones de detección de zonas homogéneas y limitación de sobrepasos

Características:
- Predicción vectorizada con procesamiento paralelo
- Ensemble optimizado para GPU
- Detección de capas acelerada
- NUEVO: Detección de zonas homogéneas y limitador de sobrepasos
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy import ndimage, signal

# Dispositivo global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUEVA FUNCIÓN: Detectar zonas homogéneas
def detect_homogeneous_zones(seismic_data, window_size=20, variance_threshold=0.05):
    """
    Detecta zonas homogéneas basándose en la varianza local de los datos sísmicos
    
    Args:
        seismic_data: Datos sísmicos
        window_size: Tamaño de ventana para cálculo de varianza
        variance_threshold: Umbral de varianza para considerar zona homogénea
        
    Returns:
        mask: Array booleano indicando zonas homogéneas (True = homogéneo)
    """
    # Convertir a numpy si es tensor
    if isinstance(seismic_data, torch.Tensor):
        seismic_data = seismic_data.cpu().numpy()
    
    # Calcular varianza local
    local_variance = np.zeros_like(seismic_data)
    half_window = window_size // 2
    
    # Cálculo vectorizado de varianza local
    for i in range(len(seismic_data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(seismic_data), i + half_window + 1)
        window_data = seismic_data[start_idx:end_idx]
        local_variance[i] = np.var(window_data)
    
    # Normalizar varianza
    if np.max(local_variance) > 0:
        normalized_variance = local_variance / np.max(local_variance)
    else:
        normalized_variance = local_variance
    
    # Crear máscara de zonas homogéneas
    homogeneous_mask = normalized_variance < variance_threshold
    
    # Dilatar ligeramente para conectar regiones cercanas
    homogeneous_mask = ndimage.binary_dilation(homogeneous_mask, iterations=2)
    
    return homogeneous_mask

# NUEVA FUNCIÓN: Limitador de sobrepasos
def limit_overshoots(predicted_impedance, real_seismic, limiter_factor=0.8, context_window=10):
    """
    Limita los sobrepasos en cambios abruptos usando información contextual
    
    Args:
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales para contexto
        limiter_factor: Factor de limitación (0-1)
        context_window: Ventana para análisis contextual
        
    Returns:
        limited_impedance: Impedancia con sobrepasos limitados
    """
    # Convertir a numpy
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    if isinstance(real_seismic, torch.Tensor):
        real_seismic = real_seismic.cpu().numpy()
    
    limited_impedance = predicted_impedance.copy()
    
    # Detectar cambios abruptos en la predicción
    gradient = np.abs(np.gradient(predicted_impedance))
    threshold = np.percentile(gradient, 95)
    abrupt_changes = gradient > threshold
    
    # Para cada cambio abrupto, verificar si es consistente con los datos sísmicos
    change_indices = np.where(abrupt_changes)[0]
    
    for idx in change_indices:
        if idx < context_window or idx >= len(predicted_impedance) - context_window:
            continue
            
        # Analizar contexto antes y después del cambio
        before_start = max(0, idx - context_window)
        before_end = idx
        after_start = idx + 1
        after_end = min(len(predicted_impedance), idx + context_window + 1)
        
        # Valores medios antes y después
        mean_before = np.mean(predicted_impedance[before_start:before_end])
        mean_after = np.mean(predicted_impedance[after_start:after_end])
        
        # Cambio predicho
        predicted_change = predicted_impedance[idx] - mean_before
        expected_change = mean_after - mean_before
        
        # Si el cambio es excesivo comparado con el contexto
        if abs(predicted_change) > abs(expected_change) * 1.5:
            # Limitar el cambio
            limited_change = expected_change + (predicted_change - expected_change) * limiter_factor
            limited_impedance[idx] = mean_before + limited_change
            
            # Suavizar transición
            if idx > 0:
                limited_impedance[idx-1] = 0.7 * limited_impedance[idx-1] + 0.3 * mean_before
            if idx < len(limited_impedance) - 1:
                limited_impedance[idx+1] = 0.7 * limited_impedance[idx+1] + 0.3 * mean_after
    
    return limited_impedance

def interpolate_missing_values(prediction, valid_indices):
    """
    Interpola valores faltantes en la predicción (version optimizada)
    
    Args:
        prediction: Tensor de predicciones
        valid_indices: Tensor booleano de índices válidos
        
    Returns:
        Tensor con valores interpolados
    """
    result = prediction.clone()
    
    # Encontrar índices válidos e inválidos
    valid_idx = torch.where(valid_indices)[0]
    invalid_idx = torch.where(~valid_indices)[0]
    
    if len(valid_idx) == 0 or len(invalid_idx) == 0:
        return result
    
    # Para cada índice inválido
    for idx in invalid_idx:
        # Encontrar índices válidos más cercanos (vectorizado)
        left_valid = valid_idx[valid_idx < idx]
        right_valid = valid_idx[valid_idx > idx]
        
        if len(left_valid) > 0 and len(right_valid) > 0:
            # Interpolar entre valores válidos a ambos lados
            left_idx = left_valid[-1]
            right_idx = right_valid[0]
            weight = (idx - left_idx).float() / (right_idx - left_idx).float()
            result[idx] = result[left_idx] * (1-weight) + result[right_idx] * weight
        elif len(left_valid) > 0:
            # Extender último valor válido a la izquierda
            result[idx] = result[left_valid[-1]]
        elif len(right_valid) > 0:
            # Extender primer valor válido a la derecha
            result[idx] = result[right_valid[0]]
    
    return result

@torch.jit.script
def gaussian_smooth_tensor(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Aplica suavizado gaussiano a un tensor (optimizado con JIT)
    
    Args:
        x: Tensor de entrada [length]
        sigma: Desviación estándar del kernel gaussiano
        
    Returns:
        Tensor suavizado
    """
    # Solo funciona para tensores 1D
    if x.dim() != 1:
        return x
    
    # Tamaño del kernel (impar)
    kernel_size = max(3, int(6 * sigma)) // 2 * 2 + 1
    
    # Crear kernel gaussiano
    kernel_range = torch.arange(kernel_size, device=x.device) - (kernel_size // 2)
    kernel = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalizar
    
    # Añadir dimensiones para conv1d [out_channels, in_channels, kernel_size]
    kernel = kernel.view(1, 1, kernel_size)
    
    # Preparar entrada para conv1d [batch, channels, length]
    x_reshaped = x.view(1, 1, -1)
    
    # Suavizado con padding de replicación
    padding = kernel_size // 2
    smoothed = F.conv1d(
        F.pad(x_reshaped, (padding, padding), mode='replicate'),
        kernel
    )
    
    # Volver a la forma original
    return smoothed.view(-1)
    
# Versión mejorada de la predicción con ensemble
def improved_ensemble_predict(models, normalized_test_seismic, window_size, window_weights=None, 
                             well_idx=0, stride_factor=6, config=None):
    """
    Predicción optimizada con ensemble para GPU - Con detección de zonas homogéneas
    
    Args:
        models: Lista de modelos entrenados
        normalized_test_seismic: Datos sísmicos normalizados
        window_size: Tamaño de la ventana
        window_weights: Pesos de la ventana (opcional)
        well_idx: Índice del pozo para usar el scaler correcto
        stride_factor: Factor para controlar solapamiento
        config: Configuración con nuevos parámetros
    
    Returns:
        Predicción combinada optimizada
    """
    # Obtener parámetros de configuración
    if config is None:
        config = {}
    homogeneous_threshold = config.get('homogeneous_zone_threshold', 0.05)
    local_variance_window = config.get('local_variance_window', 20)
    
    # Detectar zonas homogéneas ANTES de la predicción
    homogeneous_zones = detect_homogeneous_zones(
        normalized_test_seismic, 
        window_size=local_variance_window,
        variance_threshold=homogeneous_threshold
    )
    
    # Convertir a tensor y enviar al dispositivo correcto
    if not isinstance(normalized_test_seismic, torch.Tensor):
        test_seismic = torch.tensor(normalized_test_seismic, dtype=torch.float32).to(device)
    else:
        test_seismic = normalized_test_seismic.to(device=device, dtype=torch.float32)
    
    # Crear ventanas ponderadas si no se proporcionan
    if window_weights is None:
        window_weights = 0.5 * (1 - torch.cos(2 * torch.pi * torch.arange(window_size, dtype=torch.float32) / (window_size - 1)))
        window_weights = window_weights.to(device)
    elif not isinstance(window_weights, torch.Tensor):
        window_weights = torch.tensor(window_weights, dtype=torch.float32).to(device)
    else:
        window_weights = window_weights.to(device=device, dtype=torch.float32)
    
    # Configurar stride para mayor solapamiento
    stride = max(1, window_size // stride_factor)
    
    # Crear tensor de índice de pozo
    well_indices = torch.tensor([well_idx], dtype=torch.long).to(device)
    
    # Batch size para procesar ventanas (optimizar memoria GPU)
    batch_size = 32
    
    print(f"Generando predicciones con {len(models)} modelos en el ensemble para pozo {well_idx}...")
    print(f"Zonas homogéneas detectadas: {np.sum(homogeneous_zones)} de {len(homogeneous_zones)} muestras")
    
    # Predicciones de cada modelo
    all_model_predictions = []
    
    for model_idx, model in enumerate(models):
        print(f"Generando predicciones con modelo {model_idx+1}/{len(models)}...")
        model.eval()
        
        # Predicciones de este modelo
        model_acc = torch.zeros(len(test_seismic), device=device, dtype=torch.float32)
        weight_sum = torch.zeros(len(test_seismic), device=device, dtype=torch.float32)
        
        with torch.no_grad():
            # Optimización: preparar todas las ventanas para procesamiento en batch
            windows = []
            center_indices = []
            
            # Crear ventanas deslizantes - pre-procesamiento
            for i in range(0, len(test_seismic) - window_size + 1, stride):
                window = test_seismic[i:i+window_size]
                weighted_window = window * window_weights
                windows.append(weighted_window)
                
                center_idx = i + window_size // 2
                center_indices.append(center_idx)
            
            # Convertir a batch tensor
            if windows:
                windows_tensor = torch.stack(windows)
                
                # Procesar en mini-batches para evitar OOM
                num_windows = len(windows)
                for batch_start in range(0, num_windows, batch_size):
                    batch_end = min(batch_start + batch_size, num_windows)
                    batch_windows = windows_tensor[batch_start:batch_end]
                    batch_centers = center_indices[batch_start:batch_end]
                    
                    batch_windows = batch_windows.to(device=device, dtype=torch.float32)
                    
                    # Predicción con interfaz consistente
                    try:
                        batch_preds = model(batch_windows, well_indices).squeeze(-1)
                        batch_preds = batch_preds.to(dtype=torch.float32)
                        
                        # Usar ventana gaussiana para contribución ponderada
                        radius = window_size // 8
                        
                        # Para cada predicción, contribuir a múltiples posiciones
                        for j, (pred, center_idx) in enumerate(zip(batch_preds, batch_centers)):
                            start_pos = max(0, center_idx - radius)
                            end_pos = min(len(model_acc), center_idx + radius + 1)
                            positions = torch.arange(start_pos, end_pos, device=device)
                            
                            # Calcular peso gaussiano centrado en el punto central
                            dist = torch.abs(positions - center_idx).float()
                            weights = torch.exp(-0.5 * (dist / (radius/2))**2)
                            
                            # Aplicar peso adicional en zonas homogéneas
                            # Aumentar ponderación en zonas homogéneas para más estabilidad
                            if homogeneous_zones[center_idx]:
                                weights *= 1.5
                            
                            # Actualizar acumuladores
                            model_acc[start_pos:end_pos] += pred * weights
                            weight_sum[start_pos:end_pos] += weights
                    
                    except Exception as e:
                        print(f"Error en procesamiento de batch: {e}")
                        continue
        
        # Normalizar por pesos acumulados
        valid_indices = weight_sum > 0
        model_acc[valid_indices] /= weight_sum[valid_indices]
        
        # Interpolación para puntos no predichos (vectorizada)
        if torch.any(~valid_indices):
            model_acc = interpolate_missing_values(model_acc, valid_indices)
        
        # Transferir a CPU y convertir a numpy para procesamiento posterior
        all_model_predictions.append(model_acc.cpu().numpy().astype(np.float32))
    
    if not all_model_predictions:
        return np.zeros_like(normalized_test_seismic, dtype=np.float32)
    
    # Combinar predicciones del ensemble con método mejorado
    ensemble_prediction = combine_ensemble_predictions_improved(
        all_model_predictions, 
        normalized_test_seismic,
        homogeneous_zones,
        config
    )
    
    # Aplicar suavizado adaptativo preservando bordes
    final_prediction = adaptive_edge_preserving_smooth_improved(
        ensemble_prediction, 
        normalized_test_seismic,
        homogeneous_zones,
        config
    )
    
    return final_prediction

# FUNCIÓN MEJORADA: Combinar predicciones del ensemble
def combine_ensemble_predictions_improved(all_predictions, seismic_data, homogeneous_zones, config=None):
    """
    Combina predicciones del ensemble con robustez mejorada y consideración de zonas
    
    Args:
        all_predictions: Lista de predicciones de diferentes modelos
        seismic_data: Datos sísmicos normalizados
        homogeneous_zones: Máscara de zonas homogéneas
        config: Configuración con parámetros
        
    Returns:
        Predicción combinada optimizada
    """
    if config is None:
        config = {}
    outlier_threshold = config.get('ensemble_outlier_threshold', 2.0)
    
    # Convertir a numpy para operaciones estadísticas
    if isinstance(seismic_data, torch.Tensor):
        seismic_data = seismic_data.cpu().numpy()
    
    num_models = len(all_predictions)
    prediction_length = len(all_predictions[0])
    
    # Convertir lista a array para operaciones vectorizadas
    predictions_array = np.array(all_predictions)
    
    # Resultado combinado
    combined = np.zeros(prediction_length)
    
    # Análisis punto por punto con consideración de zonas
    for i in range(prediction_length):
        # Extraer valores de todos los modelos para este punto
        values = predictions_array[:, i]
        
        # En zonas homogéneas, ser más conservador
        if homogeneous_zones[i]:
            # Usar mediana robusta en zonas homogéneas
            combined[i] = np.median(values)
        else:
            # En zonas de transición, usar MAD más estricto
            median_val = np.median(values)
            abs_dev = np.abs(values - median_val)
            mad = np.median(abs_dev)
            
            if mad > 0:
                # Z-score modificado más estricto
                z_scores = 0.6745 * abs_dev / mad
                mask = z_scores < outlier_threshold
                
                # Usar solo valores no-outliers si hay suficientes
                if np.sum(mask) >= num_models/2:
                    # Media ponderada por proximidad a la mediana
                    weights = np.exp(-z_scores[mask])
                    combined[i] = np.average(values[mask], weights=weights)
                else:
                    combined[i] = median_val
            else:
                combined[i] = median_val
    
    # Post-procesamiento específico para zonas homogéneas
    # Aplicar suavizado adicional en zonas homogéneas
    for i in range(1, prediction_length - 1):
        if homogeneous_zones[i]:
            # Promedio local para estabilizar
            combined[i] = 0.5 * combined[i] + 0.25 * (combined[i-1] + combined[i+1])
    
    return combined

# Versión simplificada y optimizada para casos de error
def simple_ensemble_predict(models, normalized_test_seismic, window_size, well_idx=0, config=None):
    """
    Versión simplificada y optimizada del ensemble para casos de error
    
    Args:
        models: Lista de modelos entrenados
        normalized_test_seismic: Datos sísmicos normalizados
        window_size: Tamaño de la ventana
        well_idx: Índice del pozo
        config: Configuración con parámetros
    """
    # Detectar zonas homogéneas
    if config:
        homogeneous_zones = detect_homogeneous_zones(
            normalized_test_seismic,
            window_size=config.get('local_variance_window', 20),
            variance_threshold=config.get('homogeneous_zone_threshold', 0.05)
        )
    else:
        homogeneous_zones = np.zeros(len(normalized_test_seismic), dtype=bool)
    
    # Convertir a tensor y enviar al dispositivo correcto
    if not isinstance(normalized_test_seismic, torch.Tensor):
        test_seismic = torch.tensor(normalized_test_seismic, dtype=torch.float32).to(device)
    else:
        test_seismic = normalized_test_seismic.to(device=device, dtype=torch.float32)
    
    half_window = window_size // 2
    stride = max(1, window_size // 4)
    
    # Crear tensor de índice de pozo
    well_indices = torch.tensor([well_idx], dtype=torch.long).to(device)
    
    # Inicializar acumuladores para todas las predicciones
    all_predictions = []
    
    # Batch size para procesar ventanas
    batch_size = 32
    
    # Crear todas las ventanas de una vez para procesamiento en batch
    windows = []
    center_positions = []
    
    for i in range(half_window, len(test_seismic) - half_window, stride):
        window = test_seismic[i-half_window:i+half_window]
        windows.append(window)
        center_positions.append(i)
    
    # Convertir ventanas a tensor
    if windows:
        windows_tensor = torch.stack(windows).to(dtype=torch.float32)
    
        # Procesar con cada modelo
        for i, model in enumerate(models):
            print(f"Generando predicciones con modelo {i+1}/{len(models)} para pozo {well_idx}...")
            model.eval()
            
            # Inicializar arrays para predicciones
            model_pred = torch.zeros(len(test_seismic), device=device, dtype=torch.float32)
            weights = torch.zeros(len(test_seismic), device=device, dtype=torch.float32)
            
            with torch.no_grad():
                # Procesar en mini-batches para evitar OOM
                num_windows = len(windows)
                for batch_start in range(0, num_windows, batch_size):
                    batch_end = min(batch_start + batch_size, num_windows)
                    batch_windows = windows_tensor[batch_start:batch_end]
                    batch_centers = center_positions[batch_start:batch_end]
                    
                    try:
                        batch_windows = batch_windows.to(dtype=torch.float32)
                        
                        # Predecir con interfaz consistente
                        batch_preds = model(batch_windows, well_indices).squeeze(-1)
                        batch_preds = batch_preds.to(dtype=torch.float32)
                        
                        # Asignar predicciones a las posiciones centrales
                        for j, (pred, pos) in enumerate(zip(batch_preds, batch_centers)):
                            model_pred[pos] = pred
                            weights[pos] = 1.0
                    except Exception as e:
                        print(f"Error en predicción de batch: {e}")
                        continue
            
            # Interpolar valores faltantes
            valid_mask = weights > 0
            if torch.any(valid_mask):
                model_pred = interpolate_missing_values(model_pred, valid_mask)
                
            # Suavizado simple
            smoothed_pred = gaussian_smooth_tensor(model_pred, sigma=1.0)
                
            # Guardar predicción - asegurar que sea float32
            all_predictions.append(smoothed_pred.cpu().numpy().astype(np.float32))
    
    # Si no hay ventanas (caso extremo) o no hay predicciones
    if not windows or not all_predictions:
        return np.zeros_like(normalized_test_seismic, dtype=np.float32)
    
    # Calcular la mediana en cada punto (más robusta que la media)
    ensemble_pred = np.median(all_predictions, axis=0)
    
    # Asegurar que sea float32
    ensemble_pred = ensemble_pred.astype(np.float32)
    
    # Suavizado final adaptativo
    final_pred = adaptive_edge_preserving_smooth_improved(
        ensemble_pred, 
        normalized_test_seismic,
        homogeneous_zones,
        config
    )
    
    return final_pred
    
# FUNCIÓN MEJORADA: Suavizado adaptativo con preservación de bordes
def adaptive_edge_preserving_smooth_improved(prediction, seismic_data, homogeneous_zones, config=None):
    """
    Suavizado optimizado que preserva bordes importantes y considera zonas homogéneas
    
    Args:
        prediction: Predicción a suavizar
        seismic_data: Datos sísmicos de referencia
        homogeneous_zones: Máscara de zonas homogéneas
        config: Configuración con parámetros
        
    Returns:
        Predicción suavizada con preservación de bordes
    """
    if config is None:
        config = {}
    
    edge_threshold_percentile = config.get('edge_threshold_percentile', 94)
    overshoot_limiter = config.get('overshoot_limiter_factor', 0.8)
    
    # Asegurar que los datos son numpy arrays
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(seismic_data, torch.Tensor):
        seismic_data = seismic_data.cpu().numpy()
    
    # Primero, aplicar limitador de sobrepasos
    prediction = limit_overshoots(prediction, seismic_data, 
                                 limiter_factor=overshoot_limiter,
                                 context_window=10)
    
    # Detectar bordes usando gradiente
    gradient = np.abs(np.gradient(prediction))
    edge_threshold = np.percentile(gradient, edge_threshold_percentile)
    edge_mask = gradient > edge_threshold
    
    # Considerar reflectores sísmicos
    seismic_gradient = np.abs(np.gradient(seismic_data))
    seismic_threshold = np.percentile(seismic_gradient, 90)
    seismic_reflectors = seismic_gradient > seismic_threshold
    
    # Coincidencia entre bordes de impedancia y reflectores sísmicos
    dilated_reflectors = ndimage.binary_dilation(seismic_reflectors, iterations=3)
    confirmed_edges = edge_mask & dilated_reflectors
    
    # En zonas homogéneas, reducir detección de bordes
    edge_mask = edge_mask & ~homogeneous_zones
    edge_mask = np.logical_or(confirmed_edges, edge_mask)
    
    # Expandir máscara de bordes para proteger transiciones
    edge_mask = ndimage.binary_dilation(edge_mask, iterations=1)
    
    # Suavizado adaptativo diferenciado por zonas
    smoothed = prediction.copy()
    
    # 1. Aplicar filtro de mediana más fuerte en zonas homogéneas
    for i in range(len(prediction)):
        if homogeneous_zones[i]:
            # Ventana más grande en zonas homogéneas
            window_start = max(0, i - 5)
            window_end = min(len(prediction), i + 6)
            smoothed[i] = np.median(prediction[window_start:window_end])
    
    # 2. Suavizado gaussiano general
    smoothed = ndimage.gaussian_filter1d(smoothed, sigma=1.2)
    
    # 3. Preservar bordes mezclando original y suavizado según máscara
    edge_weight = np.zeros_like(prediction)
    edge_weight[edge_mask] = 1.0
    
    # En zonas homogéneas, preferir versión suavizada
    homogeneous_weight = homogeneous_zones.astype(float) * 0.3
    edge_weight = edge_weight * (1 - homogeneous_weight)
    
    # 4. Suavizar la máscara para transición gradual
    edge_weight = ndimage.gaussian_filter1d(edge_weight, sigma=1.0)
    
    # 5. Combinar original y suavizado
    result = prediction * edge_weight + smoothed * (1 - edge_weight)
    
    # 6. Post-procesamiento final para zonas homogéneas
    # Aplicar filtro Savitzky-Golay suave en zonas homogéneas
    for i in range(len(result)):
        if homogeneous_zones[i] and i > 10 and i < len(result) - 10:
            window_data = result[i-10:i+11]
            if len(window_data) >= 5:
                filtered = signal.savgol_filter(window_data, 5, 2)
                result[i] = filtered[10]  # Centro de la ventana
    
    return result

# Función mejorada para post-procesamiento completo
def comprehensive_impedance_improvement(predicted_impedance, real_seismic, real_impedance=None, config=None):
    """
    Post-procesamiento optimizado para mejorar la impedancia predicha
    
    Args:
        predicted_impedance: Impedancia predicha 
        real_seismic: Datos sísmicos reales
        real_impedance: Impedancia real si está disponible (opcional)
        config: Configuración con parámetros
        
    Returns:
        Impedancia mejorada con preservación de patrones
    """
    print("Aplicando mejoras avanzadas a la impedancia predicha...")
    
    if config is None:
        config = {}
    
    # Convertir a numpy si es necesario
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    if isinstance(real_seismic, torch.Tensor):
        real_seismic = real_seismic.cpu().numpy()
    if real_impedance is not None and isinstance(real_impedance, torch.Tensor):
        real_impedance = real_impedance.cpu().numpy()
    
    # Detectar zonas homogéneas
    homogeneous_zones = detect_homogeneous_zones(
        real_seismic, 
        window_size=config.get('local_variance_window', 20),
        variance_threshold=config.get('homogeneous_zone_threshold', 0.05)
    )
    
    # Paso 1: Limitar sobrepasos
    result = limit_overshoots(
        predicted_impedance, 
        real_seismic,
        limiter_factor=config.get('overshoot_limiter_factor', 0.8),
        context_window=10
    )
    
    # Paso 2: Suavizado adaptativo mejorado
    result = adaptive_edge_preserving_smooth_improved(
        result, 
        real_seismic,
        homogeneous_zones,
        config
    )
    
    # Paso 3: Alineamiento fino con reflectores sísmicos
    seismic_gradient = np.abs(np.gradient(real_seismic))
    seismic_threshold = np.percentile(seismic_gradient, 92)
    reflectors = seismic_gradient > seismic_threshold
    reflector_indices = np.where(reflectors)[0]
    
    # Ajuste fino en reflectores
    window_size = 5
    for reflector_idx in reflector_indices:
        if reflector_idx < window_size or reflector_idx >= len(result) - window_size:
            continue
            
        # Solo ajustar si NO es zona homogénea
        if not homogeneous_zones[reflector_idx]:
            # Verificar si hay un cambio de impedancia significativo cerca
            start = max(0, reflector_idx - window_size)
            end = min(len(result) - 1, reflector_idx + window_size)
            local_gradient = np.abs(np.gradient(result[start:end+1]))
            
            if np.max(local_gradient) < seismic_threshold / 3:
                # No hay cambio significativo - crear una transición suave
                if reflector_idx > window_size and reflector_idx < len(result) - window_size:
                    before_value = np.mean(result[reflector_idx-window_size:reflector_idx])
                    after_value = np.mean(result[reflector_idx+1:reflector_idx+window_size+1])
                    
                    if abs(after_value - before_value) > 0.1 * np.std(result):
                        # Crear transición suave
                        trans_start = max(0, reflector_idx - 2)
                        trans_end = min(len(result) - 1, reflector_idx + 2)
                        transition = np.linspace(before_value, after_value, trans_end - trans_start + 1)
                        result[trans_start:trans_end+1] = transition
    
    # Paso 4: Estabilización final en zonas homogéneas
    for i in range(1, len(result) - 1):
        if homogeneous_zones[i]:
            # Media móvil ponderada en zonas homogéneas
            result[i] = 0.5 * result[i] + 0.25 * (result[i-1] + result[i+1])
    
    print(f"Post-procesamiento completado. Zonas homogéneas estabilizadas: {np.sum(homogeneous_zones)}")
    
    return result

# FUNCIÓN COMPLEMENTARIA: Análisis de calidad de predicción
def analyze_prediction_quality(predicted_impedance, real_seismic, homogeneous_zones):
    """
    Analiza la calidad de la predicción por zonas
    
    Args:
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales
        homogeneous_zones: Máscara de zonas homogéneas
        
    Returns:
        Dict con métricas de calidad por zona
    """
    # Gradientes
    pred_gradient = np.abs(np.gradient(predicted_impedance))
    seismic_gradient = np.abs(np.gradient(real_seismic))
    
    # Análisis en zonas homogéneas
    homogeneous_variance = np.var(predicted_impedance[homogeneous_zones]) if np.any(homogeneous_zones) else 0
    
    # Análisis en zonas de transición
    transition_zones = ~homogeneous_zones
    transition_variance = np.var(predicted_impedance[transition_zones]) if np.any(transition_zones) else 0
    
    # Correlación de gradientes
    gradient_correlation = np.corrcoef(pred_gradient, seismic_gradient)[0, 1]
    
    # Detección de anomalías
    anomalies = pred_gradient > np.percentile(pred_gradient, 99)
    
    quality_metrics = {
        'homogeneous_stability': 1.0 / (1.0 + homogeneous_variance),
        'transition_sharpness': transition_variance,
        'gradient_correlation': gradient_correlation,
        'anomaly_count': np.sum(anomalies),
        'homogeneous_percentage': np.sum(homogeneous_zones) / len(homogeneous_zones) * 100
    }
    
    return quality_metrics

# Funciones auxiliares para la versión original (compatibilidad)
def combine_ensemble_predictions(all_predictions, seismic_data, window_size=32):
    """
    Versión original mantenida para compatibilidad
    """
    # Detectar zonas homogéneas
    homogeneous_zones = detect_homogeneous_zones(seismic_data, window_size=20)
    
    # Usar la versión mejorada
    return combine_ensemble_predictions_improved(
        all_predictions, 
        seismic_data, 
        homogeneous_zones,
        config={'ensemble_outlier_threshold': 2.5}
    )

def adaptive_edge_preserving_smooth(prediction, seismic_data, smooth_sigma=1.8, 
                                   edge_threshold_percentile=97):
    """
    Versión original mantenida para compatibilidad
    """
    # Detectar zonas homogéneas
    homogeneous_zones = detect_homogeneous_zones(seismic_data, window_size=20)
    
    # Usar la versión mejorada
    config = {
        'edge_threshold_percentile': edge_threshold_percentile,
        'overshoot_limiter_factor': 0.8
    }
    
    return adaptive_edge_preserving_smooth_improved(
        prediction, 
        seismic_data, 
        homogeneous_zones,
        config
    )

# Función de utilidad para debug
def plot_prediction_analysis(predicted_impedance, real_seismic, homogeneous_zones, save_path=None):
    """
    Visualiza el análisis de predicción con zonas homogéneas
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Impedancia predicha con zonas
    ax1 = axes[0]
    ax1.plot(predicted_impedance, 'b-', label='Impedancia Predicha')
    ax1.fill_between(np.arange(len(predicted_impedance)), 
                     np.min(predicted_impedance), np.max(predicted_impedance),
                     where=homogeneous_zones, alpha=0.3, color='green',
                     label='Zonas Homogéneas')
    ax1.set_ylabel('Impedancia')
    ax1.set_title('Impedancia Predicha con Zonas Homogéneas')
    ax1.legend()
    ax1.grid(True)
    
    # Datos sísmicos
    ax2 = axes[1]
    ax2.plot(real_seismic, 'k-', label='Datos Sísmicos')
    ax2.set_ylabel('Amplitud')
    ax2.set_title('Datos Sísmicos')
    ax2.grid(True)
    
    # Gradientes
    ax3 = axes[2]
    pred_gradient = np.abs(np.gradient(predicted_impedance))
    seismic_gradient = np.abs(np.gradient(real_seismic))
    ax3.plot(pred_gradient / np.max(pred_gradient), 'b-', label='Gradiente Impedancia (norm)')
    ax3.plot(seismic_gradient / np.max(seismic_gradient), 'r-', alpha=0.7, label='Gradiente Sísmico (norm)')
    ax3.set_ylabel('Gradiente Normalizado')
    ax3.set_xlabel('Muestra')
    ax3.set_title('Comparación de Gradientes')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
def enhanced_spike_removal(predicted_impedance, real_seismic, config=None):
    """
    Eliminación avanzada de spikes preservando transiciones reales
    
    Args:
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales para contexto
        config: Configuración con parámetros
        
    Returns:
        cleaned_impedance: Impedancia sin spikes
    """
    if config is None:
        config = {}
    
    # Convertir a numpy
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    if isinstance(real_seismic, torch.Tensor):
        real_seismic = real_seismic.cpu().numpy()
    
    # Parámetros
    spike_threshold = config.get('spike_detection_threshold', 2.5)
    homogeneous_threshold = config.get('homogeneous_zone_threshold', 0.05)
    
    # 1. DETECTAR ZONAS HOMOGÉNEAS
    homogeneous_zones = detect_homogeneous_zones(
        real_seismic, 
        window_size=config.get('local_variance_window', 20),
        variance_threshold=homogeneous_threshold
    )
    
    # 2. DETECTAR SPIKES USANDO ANÁLISIS ESTADÍSTICO ROBUSTO
    def detect_spikes_robust(data, threshold_factor=2.5):
        """Detecta spikes usando MAD (Median Absolute Deviation)"""
        if len(data) < 5:
            return np.zeros(len(data), dtype=bool)
        
        # Calcular segunda derivada
        second_diff = np.diff(data, n=2)
        
        # Estadísticas robustas
        median_val = np.median(np.abs(second_diff))
        mad = np.median(np.abs(second_diff - median_val))
        
        # Umbral adaptativo
        threshold = median_val + threshold_factor * mad
        
        # Detectar outliers
        spike_detection = np.abs(second_diff) > threshold
        
        # Expandir a todas las posiciones (padding)
        spike_mask = np.zeros(len(data), dtype=bool)
        spike_mask[1:-1] = spike_detection
        
        # Dilatar para capturar spikes de múltiples puntos
        spike_mask = ndimage.binary_dilation(spike_mask, iterations=1)
        
        return spike_mask
    
    spike_mask = detect_spikes_robust(predicted_impedance, spike_threshold)
    
    # 3. ANÁLISIS POR ZONAS
    result = predicted_impedance.copy()
    
    # ZONA HOMOGÉNEA: Aplicar suavizado agresivo
    if np.any(homogeneous_zones):
        # Filtro bilateral preservando estructura general
        homogeneous_indices = np.where(homogeneous_zones)[0]
        
        for i in homogeneous_indices:
            # Ventana local
            window_start = max(0, i - 10)
            window_end = min(len(result), i + 11)
            
            # Si hay spikes en esta zona, usar mediana local
            if spike_mask[i]:
                local_values = predicted_impedance[window_start:window_end]
                # Filtrar outliers extremos
                q25, q75 = np.percentile(local_values, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                filtered_values = local_values[
                    (local_values >= lower_bound) & (local_values <= upper_bound)
                ]
                
                if len(filtered_values) > 0:
                    result[i] = np.median(filtered_values)
                else:
                    result[i] = np.median(local_values)
            else:
                # Suavizado suave en zonas homogéneas sin spikes
                result[i] = 0.7 * result[i] + 0.3 * np.mean(predicted_impedance[window_start:window_end])
    
    # 4. TRANSICIONES: Procesamiento más cuidadoso
    transition_zones = ~homogeneous_zones
    transition_spikes = spike_mask & transition_zones
    
    if np.any(transition_spikes):
        # Analizar contexto sísmico para determinar si la transición es válida
        seismic_gradient = np.abs(np.gradient(real_seismic))
        seismic_threshold = np.percentile(seismic_gradient, 85)
        
        for spike_idx in np.where(transition_spikes)[0]:
            # Verificar si hay actividad sísmica que justifique el cambio
            seismic_support = seismic_gradient[spike_idx] > seismic_threshold
            
            if not seismic_support:
                # Es un spike espurio - suavizar
                window_start = max(0, spike_idx - 5)
                window_end = min(len(result), spike_idx + 6)
                
                # Usar tendencia local
                if window_start < spike_idx and spike_idx < window_end - 1:
                    before_trend = np.mean(predicted_impedance[window_start:spike_idx])
                    after_trend = np.mean(predicted_impedance[spike_idx+1:window_end])
                    result[spike_idx] = 0.6 * before_trend + 0.4 * after_trend
    
    # 5. SUAVIZADO FINAL ADAPTATIVO
    # Aplicar filtro Savitzky-Golay con ventana adaptativa
    for i in range(len(result)):
        if homogeneous_zones[i]:
            # Ventana más grande en zonas homogéneas
            window_size = min(15, len(result) // 4)
        else:
            # Ventana más pequeña en transiciones
            window_size = min(7, len(result) // 8)
        
        # Asegurar ventana impar y mínima
        window_size = max(5, window_size)
        if window_size % 2 == 0:
            window_size += 1
        
        # Aplicar solo si tenemos suficientes puntos
        if i >= window_size//2 and i < len(result) - window_size//2:
            start_idx = i - window_size//2
            end_idx = i + window_size//2 + 1
            local_data = result[start_idx:end_idx]
            
            if len(local_data) >= window_size:
                try:
                    filtered = signal.savgol_filter(local_data, window_size, 3)
                    result[i] = filtered[window_size//2]
                except:
                    pass  # Mantener valor original si falla el filtro
    
    return result


def enhanced_high_value_recovery(predicted_impedance, real_impedance, config=None):
    """
    Mejora la predicción en regiones de valores altos
    
    Args:
        predicted_impedance: Impedancia predicha
        real_impedance: Impedancia real (para referencia)
        config: Configuración
        
    Returns:
        enhanced_impedance: Impedancia con valores altos mejorados
    """
    if config is None:
        config = {}
    
    # Convertir a numpy
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    if isinstance(real_impedance, torch.Tensor):
        real_impedance = real_impedance.cpu().numpy()
    
    # Parámetros
    high_threshold = config.get('dynamic_scaling_threshold', 50000)
    boost_factor = config.get('high_value_boost', 1.2)
    
    result = predicted_impedance.copy()
    
    # Identificar regiones donde el valor real es alto pero la predicción es baja
    high_value_regions = real_impedance > high_threshold
    underestimated = (predicted_impedance < real_impedance * 0.8) & high_value_regions
    
    if np.any(underestimated):
        # Aplicar boost progresivo
        for i in np.where(underestimated)[0]:
            # Calcular factor de boost basado en la diferencia
            target_ratio = real_impedance[i] / predicted_impedance[i]
            adjusted_boost = min(boost_factor, 1.0 + (target_ratio - 1.0) * 0.5)
            
            result[i] = predicted_impedance[i] * adjusted_boost
            
            # Suavizar la transición en puntos adyacentes
            for offset in [-2, -1, 1, 2]:
                adj_idx = i + offset
                if 0 <= adj_idx < len(result):
                    distance_factor = 1.0 - abs(offset) * 0.2
                    transition_boost = 1.0 + (adjusted_boost - 1.0) * distance_factor
                    result[adj_idx] = predicted_impedance[adj_idx] * transition_boost
    
    return result


def enhanced_transition_smoothing(predicted_impedance, real_seismic, homogeneous_zones, config=None):
    """
    Suavizado específico para transiciones bruscas
    
    Args:
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales
        homogeneous_zones: Máscara de zonas homogéneas
        config: Configuración
        
    Returns:
        smoothed_impedance: Impedancia con transiciones suavizadas
    """
    if config is None:
        config = {}
    
    result = predicted_impedance.copy()
    
    # Detectar transiciones bruscas
    gradient = np.abs(np.gradient(predicted_impedance))
    transition_threshold = np.percentile(gradient, 90)
    abrupt_transitions = gradient > transition_threshold
    
    # Detectar reflectores sísmicos importantes
    seismic_gradient = np.abs(np.gradient(real_seismic))
    seismic_reflector_threshold = np.percentile(seismic_gradient, 88)
    seismic_reflectors = seismic_gradient > seismic_reflector_threshold
    
    # Suavizar transiciones que NO están respaldadas por actividad sísmica
    for i in np.where(abrupt_transitions)[0]:
        # Solo suavizar si no es zona homogénea y no hay reflector sísmico
        if not homogeneous_zones[i] and not seismic_reflectors[i]:
            # Verificar ventana local para determinar si la transición es espuria
            window_start = max(0, i - 3)
            window_end = min(len(result), i + 4)
            
            # Analizar consistencia local
            local_values = predicted_impedance[window_start:window_end]
            median_local = np.median(local_values)
            
            # Si el valor actual es muy diferente de la mediana local
            if abs(predicted_impedance[i] - median_local) > np.std(local_values) * 2:
                # Crear transición más suave
                if i > 0 and i < len(result) - 1:
                    # Interpolación cúbica local
                    x_points = [i-1, i+1]
                    y_points = [predicted_impedance[i-1], predicted_impedance[i+1]]
                    
                    # Valor interpolado
                    interpolated = np.interp(i, x_points, y_points)
                    
                    # Mezclar con valor original (preservar algo de la transición)
                    result[i] = 0.4 * predicted_impedance[i] + 0.6 * interpolated
    
    return result


def comprehensive_impedance_enhancement(predicted_impedance, real_seismic, real_impedance=None, config=None):
    """
    Mejora integral de la impedancia predicha con todos los métodos optimizados
    
    Args:
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales
        real_impedance: Impedancia real (opcional, para calibración)
        config: Configuración con parámetros
        
    Returns:
        enhanced_impedance: Impedancia mejorada
    """
    print("Aplicando mejora integral de impedancia...")
    
    if config is None:
        config = {}
    
    # Detectar zonas homogéneas una sola vez
    homogeneous_zones = detect_homogeneous_zones(
        real_seismic,
        window_size=config.get('local_variance_window', 20),
        variance_threshold=config.get('homogeneous_zone_threshold', 0.05)
    )
    
    # Paso 1: Eliminación de spikes
    result = enhanced_spike_removal(predicted_impedance, real_seismic, config)
    print(f"  Paso 1: Eliminación de spikes completada")
    
    # Paso 2: Suavizado de transiciones bruscas
    result = enhanced_transition_smoothing(result, real_seismic, homogeneous_zones, config)
    print(f"  Paso 2: Suavizado de transiciones completada")
    
    # Paso 3: Mejora de valores altos (si tenemos referencia)
    if real_impedance is not None:
        result = enhanced_high_value_recovery(result, real_impedance, config)
        print(f"  Paso 3: Mejora de valores altos completada")
    
    # Paso 4: Limitación de sobrepasos (del código original)
    result = limit_overshoots(result, real_seismic, 
                             limiter_factor=config.get('overshoot_limiter_factor', 0.8),
                             context_window=8)
    print(f"  Paso 4: Limitación de sobrepasos completada")
    
    # Paso 5: Suavizado final preservando bordes importantes
    result = adaptive_edge_preserving_smooth_improved(
        result, 
        real_seismic,
        homogeneous_zones,
        config
    )
    print(f"  Paso 5: Suavizado final completado")
    
    # Paso 6: Estabilización final en zonas homogéneas
    for i in range(1, len(result) - 1):
        if homogeneous_zones[i]:
            # Aplicar filtro de media móvil ponderada más agresivo
            window_start = max(0, i - 7)
            window_end = min(len(result), i + 8)
            local_values = result[window_start:window_end]
            
            # Usar mediana para mayor robustez
            result[i] = 0.4 * result[i] + 0.6 * np.median(local_values)
    
    print(f"  Paso 6: Estabilización en zonas homogéneas completada")
    print(f"Mejora integral completada. Zonas homogéneas: {np.sum(homogeneous_zones)}")
    
    return result


def adaptive_bilateral_filter(predicted_impedance, homogeneous_zones, sigma_spatial=5, sigma_intensity=10000):
    """
    Filtro bilateral adaptativo que preserva bordes importantes
    
    Args:
        predicted_impedance: Impedancia a filtrar
        homogeneous_zones: Máscara de zonas homogéneas
        sigma_spatial: Parámetro espacial del filtro
        sigma_intensity: Parámetro de intensidad del filtro
        
    Returns:
        filtered_impedance: Impedancia filtrada
    """
    result = predicted_impedance.copy()
    
    for i in range(len(predicted_impedance)):
        if homogeneous_zones[i]:
            # En zonas homogéneas, aplicar filtro más agresivo
            window_start = max(0, i - sigma_spatial)
            window_end = min(len(predicted_impedance), i + sigma_spatial + 1)
            
            # Pesos espaciales
            spatial_weights = np.exp(-0.5 * ((np.arange(window_start, window_end) - i) / sigma_spatial) ** 2)
            
            # Pesos de intensidad
            intensity_diff = np.abs(predicted_impedance[window_start:window_end] - predicted_impedance[i])
            intensity_weights = np.exp(-0.5 * (intensity_diff / sigma_intensity) ** 2)
            
            # Combinar pesos
            combined_weights = spatial_weights * intensity_weights
            combined_weights /= np.sum(combined_weights)
            
            # Valor filtrado
            result[i] = np.sum(predicted_impedance[window_start:window_end] * combined_weights)
    
    return result


# FUNCIÓN PRINCIPAL PARA REEMPLAZAR EN EL CÓDIGO EXISTENTE
def improved_ensemble_predict_with_enhancement(models, normalized_test_seismic, window_size, 
                                             window_weights=None, well_idx=0, stride_factor=6, config=None):
    """
    Versión mejorada de la predicción con ensemble que incluye todas las mejoras
    Esta función debe reemplazar la función improved_ensemble_predict en PINN_1D_OPTIMIZADO.py
    """
    # Usar la función original para obtener la predicción base
    from PINN_1D_OPTIMIZADO import improved_ensemble_predict
    
    # Obtener predicción base
    base_prediction = improved_ensemble_predict(
        models, normalized_test_seismic, window_size, window_weights, 
        well_idx, stride_factor, config
    )
    
    # Aplicar mejoras avanzadas
    enhanced_prediction = comprehensive_impedance_enhancement(
        base_prediction, 
        normalized_test_seismic, 
        real_impedance=None,  # No tenemos la referencia en predicción
        config=config
    )
    
    return enhanced_prediction
