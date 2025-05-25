"""
PINN_1B_OPTIMIZADO.py - Arquitectura del Modelo Optimizada y Corregida
PARTE 1: Arquitectura base y bloques residuales

Características:
- Arquitectura multiescala completamente vectorizada
- Optimizada para aprovechamiento máximo de GPU
- Interfaz consistente para índices de pozo
- Corregido el problema con operaciones inplace
- MEJORADO: Función de pérdida con penalización adaptativa por zonas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Habilitar optimizaciones de cuDNN si está disponible
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Clase para bloque residual optimizado
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.15):
        super(ResidualBlock, self).__init__()
        
        # Secuencial principal con módulos optimizados
        # CORRECCIÓN: Eliminados los inplace=True para evitar problemas con AMP
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(negative_slope=0.01),  # Eliminado inplace=True
            nn.Dropout(dropout_rate),           # Eliminado inplace=True
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Proyección para el atajo si las dimensiones difieren (pre-compilada)
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.final_activation = nn.LeakyReLU(negative_slope=0.01)  # Eliminado inplace=True
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out = out + identity  # Conexión residual
        return self.final_activation(out)

# Modelo multiescala optimizado para GPU
class MultiScaleImpedanceModel(nn.Module):
    def __init__(self, input_size, hidden_size=384, dropout_rate=0.15, output_size=1):
        super(MultiScaleImpedanceModel, self).__init__()
        
        # Definir las escalas (mantenemos 3 escalas)
        self.scales = [
            {"size": input_size, "stride": 1},      # Escala completa
            {"size": input_size//2, "stride": 2},   # Media escala
            {"size": input_size//4, "stride": 4}    # Cuarto de escala
        ]
        
        # Pooling layers precompilados para optimización
        self.pools = nn.ModuleList([
            nn.Identity(),  # Para escala completa no se necesita pooling
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.AvgPool1d(kernel_size=4, stride=4)
        ])
        
        # Encoders para cada escala
        # CORRECCIÓN: Eliminados los inplace=True para evitar problemas con AMP
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(scale["size"], hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(negative_slope=0.01),  # Eliminado inplace=True
                nn.Dropout(dropout_rate)            # Eliminado inplace=True
            ) for scale in self.scales
        ])
        
        # Fusión de características
        fusion_size = hidden_size
        # CORRECCIÓN: Eliminados los inplace=True para evitar problemas con AMP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * len(self.scales), fusion_size),
            nn.BatchNorm1d(fusion_size),
            nn.LeakyReLU(negative_slope=0.01),  # Eliminado inplace=True
            nn.Dropout(dropout_rate)            # Eliminado inplace=True
        )
        
        # Bloques residuales para mejor flujo de gradientes
        self.res_blocks = nn.Sequential(
            ResidualBlock(fusion_size, fusion_size, dropout_rate),
            ResidualBlock(fusion_size, fusion_size, dropout_rate),
            ResidualBlock(fusion_size, fusion_size, dropout_rate)
        )
        
        # Capa de salida con mayor reducción gradual
        # CORRECCIÓN: Eliminados los inplace=True para evitar problemas con AMP
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_size, fusion_size//2),
            nn.LeakyReLU(negative_slope=0.01),  # Eliminado inplace=True
            nn.Linear(fusion_size//2, fusion_size//4),
            nn.LeakyReLU(negative_slope=0.01),  # Eliminado inplace=True
            nn.Linear(fusion_size//4, output_size)
        )
        
        # Inicialización de pesos mejorada
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicialización optimizada de pesos para convergencia más rápida"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x, well_idx=None):
        """
        Forward pass vectorizado
        
        Args:
            x: Datos sísmicos normalizados [batch_size, window_size]
            well_idx: Índice del pozo (opcional, para interfaz consistente)
                     
        Returns:
            Impedancia acústica predicha [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # Extraer características a diferentes escalas (paralelizado)
        multi_scale_features = []
        
        for i, (pool, encoder) in enumerate(zip(self.pools, self.encoders)):
            # Realizar pooling si es necesario
            if i > 0:  # No es la escala original
                # Reshape para pooling [batch, channel, length]
                x_reshaped = x.view(batch_size, 1, -1)
                # Aplicar pooling
                x_pooled = pool(x_reshaped)
                # Aplanar de nuevo [batch, length]
                x_pooled = x_pooled.view(batch_size, -1)
                
                # Asegurar dimensión correcta
                target_size = self.scales[i]["size"]
                if x_pooled.size(1) < target_size:
                    # Padding con ceros si es necesario
                    padding = torch.zeros(batch_size, target_size - x_pooled.size(1), 
                                         dtype=x_pooled.dtype, device=x_pooled.device)
                    x_pooled = torch.cat([x_pooled, padding], dim=1)
                elif x_pooled.size(1) > target_size:
                    # Truncar si es necesario
                    x_pooled = x_pooled[:, :target_size]
                
                # Procesar con el encoder
                features = encoder(x_pooled)
            else:
                # Para la escala original, usar directamente
                features = encoder(x)
                
            multi_scale_features.append(features)
        
        # Concatenar características (más eficiente que append iterativo)
        combined = torch.cat(multi_scale_features, dim=1)
        
        # Fusión y procesamiento
        fused = self.fusion(combined)
        processed = self.res_blocks(fused)
        output = self.output_layer(processed)
        
        return output

# FUNCIÓN DE PÉRDIDA MEJORADA: Con penalización adaptativa por zonas
class AdaptiveGradientConstraintLoss(nn.Module):
    def __init__(self, max_gradient_threshold=0.15, gradient_weight=0.1, 
                 homogeneous_penalty=2.0, transition_penalty=0.5):
        """
        Función de pérdida con restricción de gradientes adaptativa por zonas
        
        Args:
            max_gradient_threshold: Umbral máximo de gradiente permitido
            gradient_weight: Peso base para la pérdida de gradiente
            homogeneous_penalty: Factor de penalización en zonas homogéneas
            transition_penalty: Factor de penalización en zonas de transición
        """
        super(AdaptiveGradientConstraintLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.max_gradient_threshold = max_gradient_threshold
        self.gradient_weight = gradient_weight
        self.homogeneous_penalty = homogeneous_penalty
        self.transition_penalty = transition_penalty
        
    def detect_zones(self, data, window_size=10, variance_threshold=0.1):
        """
        Detecta zonas homogéneas vs transiciones en los datos
        
        Args:
            data: Tensor de datos [batch_size, seq_len]
            window_size: Tamaño de ventana para análisis
            variance_threshold: Umbral para considerar zona homogénea
            
        Returns:
            homogeneous_mask: Máscara booleana de zonas homogéneas
        """
        batch_size, seq_len = data.shape
        
        if seq_len <= window_size:
            # Si la secuencia es muy corta, considerarla toda como transición
            return torch.zeros_like(data, dtype=torch.bool)
        
        # Calcular varianza local usando unfold para eficiencia
        unfolded = data.unfold(1, window_size, 1)  # [batch, windows, window_size]
        local_variance = unfolded.var(dim=2)  # [batch, windows]
        
        # Pad para mantener dimensiones originales
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        padded_variance = F.pad(local_variance, (pad_left, pad_right), mode='replicate')
        
        # Normalizar varianza por batch
        max_variance = padded_variance.max(dim=1, keepdim=True)[0]
        normalized_variance = padded_variance / (max_variance + 1e-8)
        
        # Crear máscara de zonas homogéneas
        homogeneous_mask = normalized_variance < variance_threshold
        
        return homogeneous_mask
        
    def forward(self, impedance_pred, impedance_true):
        """
        Función de pérdida con restricción de gradientes adaptativa
        
        Args:
            impedance_pred: Predicciones del modelo [batch_size, seq_len]
            impedance_true: Valores reales [batch_size, seq_len]
            
        Returns:
            loss: Pérdida total combinada
        """
        # MSE regular entre predicciones y valores reales
        mse_loss = self.mse(impedance_pred, impedance_true)
        
        # Tensor dimensions
        batch_size = impedance_pred.shape[0]
        seq_len = impedance_pred.shape[1] if len(impedance_pred.shape) > 1 else 1
        
        # Para salidas escalares, solo podemos usar MSE
        if seq_len == 1:
            return mse_loss
        
        # Detectar zonas homogéneas en los datos reales
        homogeneous_zones = self.detect_zones(impedance_true)
        
        # Calcular gradientes
        true_diffs = torch.abs(impedance_true[:, 1:] - impedance_true[:, :-1])
        pred_diffs = torch.abs(impedance_pred[:, 1:] - impedance_pred[:, :-1])
        
        # Pérdida de similitud de patrones (preservar tendencias)
        pattern_loss = torch.mean((pred_diffs - true_diffs) ** 2)
        
        # Pérdida adaptativa por zonas
        zone_masks = homogeneous_zones[:, 1:]  # Ajustar para coincidir con diffs
        
        # En zonas homogéneas: penalizar fuertemente cualquier gradiente
        homogeneous_gradients = pred_diffs * zone_masks.float()
        homogeneous_loss = torch.mean(homogeneous_gradients ** 2) * self.homogeneous_penalty
        
        # En zonas de transición: permitir gradientes pero con límite
        transition_masks = ~zone_masks
        transition_gradients = pred_diffs * transition_masks.float()
        
        # Crear umbral adaptativo basado en gradientes reales
        threshold_tensor = torch.max(
            true_diffs, 
            torch.ones_like(true_diffs) * self.max_gradient_threshold
        )
        
        # Penalizar solo excesos sobre el umbral en transiciones
        excess_gradients = F.relu(transition_gradients - threshold_tensor)
        transition_loss = torch.mean(excess_gradients ** 2) * self.transition_penalty
        
        # Combinar todas las pérdidas
        total_gradient_loss = homogeneous_loss + transition_loss
        
        # Pérdida total con pesos
        return mse_loss + pattern_loss * 0.3 + total_gradient_loss * self.gradient_weight

# Función de pérdida original mejorada (mantenida para compatibilidad)
class GradientConstraintLoss(nn.Module):
    def __init__(self, max_gradient_threshold=0.15, gradient_weight=0.1):
        super(GradientConstraintLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.max_gradient_threshold = max_gradient_threshold
        self.gradient_weight = gradient_weight
        
    def forward(self, impedance_pred, impedance_true):
        """
        Función de pérdida con restricción de gradientes optimizada
        """
        # MSE regular entre predicciones y valores reales
        mse_loss = self.mse(impedance_pred, impedance_true)
        
        # Tensor dimensions
        batch_size = impedance_pred.shape[0]
        seq_len = impedance_pred.shape[1] if len(impedance_pred.shape) > 1 else 1
        
        # Para salidas escalares, solo podemos usar MSE
        if seq_len == 1:
            return mse_loss
            
        # Calcular gradientes utilizando operaciones vectorizadas de tensor
        # Diferencia entre valores adyacentes
        true_diffs = torch.abs(impedance_true[:, 1:] - impedance_true[:, :-1])
        pred_diffs = torch.abs(impedance_pred[:, 1:] - impedance_pred[:, :-1])
        
        # Pérdida de similitud de patrones (preservar tendencias)
        pattern_loss = torch.mean((pred_diffs - true_diffs) ** 2)
        
        # Pérdida que penaliza exceso de gradiente pero preserva patrones
        # Creamos un tensor de umbral basado en los gradientes reales o un mínimo
        threshold_tensor = torch.max(
            true_diffs, 
            torch.ones_like(true_diffs) * self.max_gradient_threshold
        )
        
        # Solo penalizamos gradientes que exceden el umbral (ReLU)
        gradient_loss = torch.mean(
            torch.nn.functional.relu(pred_diffs - threshold_tensor) ** 2
        )
        
        # Combinar pérdidas con pesos optimizados
        return mse_loss + pattern_loss * 0.3 + gradient_loss * self.gradient_weight

# NUEVA CLASE: Pérdida con suavizado temporal
class TemporalSmoothingLoss(nn.Module):
    def __init__(self, smoothing_weight=0.1):
        """
        Pérdida que incentiva suavidad temporal
        
        Args:
            smoothing_weight: Peso para la pérdida de suavizado
        """
        super(TemporalSmoothingLoss, self).__init__()
        self.smoothing_weight = smoothing_weight
        
    def forward(self, predictions):
        """
        Calcula pérdida de suavizado temporal
        
        Args:
            predictions: Tensor de predicciones [batch_size, seq_len]
            
        Returns:
            smoothing_loss: Pérdida de suavizado
        """
        if predictions.shape[1] < 3:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calcular segunda derivada (aceleración)
        first_diff = predictions[:, 1:] - predictions[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        # Penalizar cambios bruscos en la aceleración
        smoothing_loss = torch.mean(second_diff ** 2)
        
        return smoothing_loss * self.smoothing_weight

# NUEVA CLASE: Pérdida combinada adaptativa
class CombinedAdaptiveLoss(nn.Module):
    def __init__(self, config):
        """
        Pérdida combinada que usa todas las técnicas adaptativas
        
        Args:
            config: Diccionario de configuración con parámetros
        """
        super(CombinedAdaptiveLoss, self).__init__()
        
        # Pérdida principal adaptativa
        self.adaptive_loss = AdaptiveGradientConstraintLoss(
            max_gradient_threshold=config.get('max_gradient_threshold', 0.12),
            gradient_weight=config.get('gradient_weight', 0.25),
            homogeneous_penalty=2.0,
            transition_penalty=0.5
        )
        
        # Pérdida de suavizado temporal
        self.temporal_loss = TemporalSmoothingLoss(
            smoothing_weight=config.get('temporal_smoothing_weight', 0.05)
        )
        
        # Pesos para combinar pérdidas
        self.temporal_weight = config.get('temporal_loss_weight', 0.1)
        
    def forward(self, impedance_pred, impedance_true):
        """
        Calcula pérdida total combinada
        
        Args:
            impedance_pred: Predicciones [batch_size, seq_len]
            impedance_true: Valores reales [batch_size, seq_len]
            
        Returns:
            total_loss: Pérdida total
            loss_components: Dict con componentes individuales de la pérdida
        """
        # Pérdida adaptativa principal
        adaptive_loss = self.adaptive_loss(impedance_pred, impedance_true)
        
        # Pérdida de suavizado temporal
        temporal_loss = self.temporal_loss(impedance_pred)
        
        # Pérdida total
        total_loss = adaptive_loss + temporal_loss * self.temporal_weight
        
        # Componentes para logging
        loss_components = {
            'adaptive': adaptive_loss.item(),
            'temporal': temporal_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
        
class EnhancedSpikeControlLoss(nn.Module):
    def __init__(self, config):
        """
        Función de pérdida mejorada para control específico de spikes
        
        Args:
            config: Diccionario de configuración con parámetros avanzados
        """
        super(EnhancedSpikeControlLoss, self).__init__()
        self.mse = nn.MSELoss()
        
        # Parámetros básicos
        self.max_gradient_threshold = config.get('max_gradient_threshold', 0.12)
        self.gradient_weight = config.get('gradient_weight', 0.25)
        
        # Parámetros avanzados para control de spikes
        self.homogeneous_penalty = config.get('homogeneous_penalty_factor', 3.0)
        self.transition_penalty = config.get('transition_penalty_factor', 0.4)
        self.spike_threshold = config.get('spike_detection_threshold', 2.5)
        self.temporal_weight = config.get('temporal_smoothing_weight', 0.08)
        
        # Parámetros para valores altos
        self.high_value_boost = config.get('high_value_boost', 1.2)
        self.scaling_threshold = config.get('dynamic_scaling_threshold', 50000)
        
    def detect_spikes(self, prediction, threshold_factor=2.5):
        """
        Detecta spikes en la predicción usando análisis estadístico
        
        Args:
            prediction: Tensor de predicciones [batch, seq_len]
            threshold_factor: Factor multiplicativo para detección
            
        Returns:
            spike_mask: Máscara booleana de spikes detectados
        """
        if prediction.shape[1] < 3:
            return torch.zeros_like(prediction, dtype=torch.bool)
        
        # Calcular segunda derivada (aceleración) para detectar cambios abruptos
        first_diff = prediction[:, 1:] - prediction[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        # Estadísticas robustas
        median_second_diff = torch.median(torch.abs(second_diff), dim=1, keepdim=True)[0]
        mad = torch.median(torch.abs(second_diff - median_second_diff), dim=1, keepdim=True)[0]
        
        # Umbral adaptativo
        threshold = median_second_diff + threshold_factor * mad
        
        # Detectar spikes (pad para mantener dimensiones)
        spike_detection = torch.abs(second_diff) > threshold
        
        # Expandir la detección de spikes a posiciones adyacentes
        spike_mask = torch.zeros_like(prediction, dtype=torch.bool)
        spike_mask[:, 1:-1] = spike_detection
        
        # Expandir spikes a posiciones vecinas
        if torch.any(spike_detection):
            dilated_spikes = F.max_pool1d(
                spike_detection.float().unsqueeze(1), 
                kernel_size=3, stride=1, padding=1
            ).squeeze(1) > 0
            spike_mask[:, 1:-1] = dilated_spikes
        
        return spike_mask
    
    def detect_homogeneous_zones_tensor(self, data, window_size=15, variance_threshold=0.05):
        """
        Detecta zonas homogéneas usando operaciones de tensor
        
        Args:
            data: Tensor de datos [batch, seq_len]
            window_size: Tamaño de ventana para análisis
            variance_threshold: Umbral de varianza normalizada
            
        Returns:
            homogeneous_mask: Máscara de zonas homogéneas
        """
        batch_size, seq_len = data.shape
        
        if seq_len <= window_size:
            return torch.zeros_like(data, dtype=torch.bool)
        
        # Calcular varianza local usando unfold
        unfolded = data.unfold(1, window_size, 1)  # [batch, windows, window_size]
        local_variance = unfolded.var(dim=2)  # [batch, windows]
        
        # Padding para mantener dimensiones originales
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        padded_variance = F.pad(local_variance, (pad_left, pad_right), mode='replicate')
        
        # Normalizar varianza por batch
        max_variance = padded_variance.max(dim=1, keepdim=True)[0]
        normalized_variance = padded_variance / (max_variance + 1e-8)
        
        # Crear máscara de zonas homogéneas
        homogeneous_mask = normalized_variance < variance_threshold
        
        return homogeneous_mask
    
    def apply_high_value_boost(self, prediction, target):
        """
        Aplica boost dinámico para valores altos de impedancia
        
        Args:
            prediction: Predicciones del modelo
            target: Valores objetivo
            
        Returns:
            boosted_prediction: Predicción con boost aplicado
        """
        # Identificar regiones de valores altos en el target
        high_value_mask = target > self.scaling_threshold
        
        if torch.any(high_value_mask):
            # Aplicar boost multiplicativo en regiones de valores altos
            boosted_prediction = prediction.clone()
            boosted_prediction[high_value_mask] *= self.high_value_boost
            
            # Suavizar la transición del boost
            transition_mask = (target > self.scaling_threshold * 0.8) & (target <= self.scaling_threshold)
            if torch.any(transition_mask):
                transition_factor = 1.0 + (self.high_value_boost - 1.0) * \
                                  ((target[transition_mask] - self.scaling_threshold * 0.8) / 
                                   (self.scaling_threshold * 0.2))
                boosted_prediction[transition_mask] *= transition_factor
            
            return boosted_prediction
        
        return prediction
    
    def forward(self, impedance_pred, impedance_true):
        """
        Función de pérdida principal con control avanzado de spikes
        
        Args:
            impedance_pred: Predicciones [batch, seq_len] 
            impedance_true: Valores reales [batch, seq_len]
            
        Returns:
            total_loss: Pérdida total con todos los componentes
        """
        # MSE base
        mse_loss = self.mse(impedance_pred, impedance_true)
        
        # Para salidas escalares, usar solo MSE
        if impedance_pred.shape[1] == 1:
            return mse_loss
        
        batch_size, seq_len = impedance_pred.shape
        
        # Aplicar boost para valores altos
        boosted_pred = self.apply_high_value_boost(impedance_pred, impedance_true)
        
        # Recalcular MSE con boost
        mse_loss = self.mse(boosted_pred, impedance_true)
        
        # Detectar zonas homogéneas y spikes
        homogeneous_zones = self.detect_homogeneous_zones_tensor(impedance_true)
        spike_mask = self.detect_spikes(boosted_pred, self.spike_threshold)
        
        # 1. PÉRDIDA POR SPIKES - Penalizar fuertemente spikes detectados
        spike_penalty = 0.0
        if torch.any(spike_mask):
            spike_values = boosted_pred[spike_mask]
            corresponding_true = impedance_true[spike_mask]
            spike_penalty = torch.mean((spike_values - corresponding_true) ** 2) * 5.0
        
        # 2. PÉRDIDA POR GRADIENTES ADAPTATIVA
        # Calcular gradientes
        pred_diffs = torch.abs(boosted_pred[:, 1:] - boosted_pred[:, :-1])
        true_diffs = torch.abs(impedance_true[:, 1:] - impedance_true[:, :-1])
        
        # Pérdida de similitud de patrones
        pattern_loss = torch.mean((pred_diffs - true_diffs) ** 2)
        
        # Umbral adaptativo basado en gradientes reales
        threshold_tensor = torch.max(
            true_diffs, 
            torch.ones_like(true_diffs) * self.max_gradient_threshold
        )
        
        # Penalización diferenciada por zonas
        zone_masks = homogeneous_zones[:, 1:]  # Ajustar para coincidir con diffs
        
        # En zonas homogéneas: penalizar MÁS cualquier gradiente
        homogeneous_gradients = pred_diffs * zone_masks.float()
        homogeneous_loss = torch.mean(homogeneous_gradients ** 2) * self.homogeneous_penalty
        
        # En zonas de transición: permitir gradientes controlados
        transition_masks = ~zone_masks
        transition_gradients = pred_diffs * transition_masks.float()
        
        # Solo penalizar excesos significativos en transiciones
        excess_gradients = F.relu(transition_gradients - threshold_tensor * 1.5)
        transition_loss = torch.mean(excess_gradients ** 2) * self.transition_penalty
        
        # 3. PÉRDIDA DE SUAVIZADO TEMPORAL
        temporal_loss = 0.0
        if seq_len > 2:
            # Segunda derivada para suavidad
            second_diff = boosted_pred[:, 2:] - 2*boosted_pred[:, 1:-1] + boosted_pred[:, :-2]
            temporal_loss = torch.mean(second_diff ** 2) * self.temporal_weight
        
        # 4. PÉRDIDA ESPECÍFICA PARA PRESERVAR TRANSICIONES REALES
        transition_preservation = 0.0
        if torch.any(~zone_masks):
            # En transiciones reales, asegurar que la predicción siga la tendencia
            transition_pred = boosted_pred[:, 1:][~zone_masks]
            transition_true = impedance_true[:, 1:][~zone_masks]
            
            if len(transition_pred) > 0:
                transition_preservation = torch.mean((transition_pred - transition_true) ** 2) * 0.5
        
        # COMBINAR TODAS LAS PÉRDIDAS
        gradient_loss = homogeneous_loss + transition_loss
        total_loss = (mse_loss + 
                     pattern_loss * 0.3 + 
                     gradient_loss * self.gradient_weight +
                     spike_penalty * 0.8 +
                     temporal_loss +
                     transition_preservation)
        
        return total_loss


class AdaptiveWindowLoss(nn.Module):
    def __init__(self, config):
        """
        Pérdida con ventanas adaptativas según el tipo de zona
        """
        super(AdaptiveWindowLoss, self).__init__()
        self.base_loss = EnhancedSpikeControlLoss(config)
        self.adaptive_windows = config.get('adaptive_window_sizing', True)
        
    def forward(self, impedance_pred, impedance_true):
        """
        Aplica pérdida con ventanas adaptativas
        """
        if not self.adaptive_windows:
            return self.base_loss(impedance_pred, impedance_true)
        
        # Para implementación completa, usar la pérdida base por ahora
        return self.base_loss(impedance_pred, impedance_true)
