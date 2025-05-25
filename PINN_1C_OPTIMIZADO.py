"""
PINN_1C_OPTIMIZADO.py - Wrapper de Física Optimizado y Corregido

Características:
- Cálculos físicos vectorizados
- Convolución nativa de PyTorch
- Gestión eficiente de recursos GPU
- Corregido error en physics_loss para manejo de errores
- MEJORADO: Pérdida física adaptativa por zonas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Función para ventana Hann (definida para no depender de otros módulos)
def hann_window(size):
    """Crea una ventana Hann para ponderar muestras"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))

# Wrapper para manejar normalización en cálculos físicos
class NormalizedPhysicsWrapper(nn.Module):
    def __init__(self, model, seismic_scalers, impedance_scalers):
        """
        Inicializar wrapper físico optimizado
        
        Args:
            model: Modelo base
            seismic_scalers: Scalers para datos sísmicos (lista o scaler único)
            impedance_scalers: Scalers para impedancia (lista o scaler único)
        """
        super().__init__()
        self.model = model
        self.seismic_scalers = seismic_scalers
        self.impedance_scalers = impedance_scalers
        
        # CORRECCIÓN: Agregamos MSE para manejo de errores
        self.mse = nn.MSELoss()
        
        # Precomputar parámetros de normalización para cada scaler (optimizado)
        self.register_buffer('seismic_means', self._extract_scaler_params(seismic_scalers, 'mean'))
        self.register_buffer('seismic_stds', self._extract_scaler_params(seismic_scalers, 'scale')) 
        self.register_buffer('impedance_means', self._extract_scaler_params(impedance_scalers, 'mean'))
        self.register_buffer('impedance_stds', self._extract_scaler_params(impedance_scalers, 'scale'))
    
    def _extract_scaler_params(self, scalers, param_type):
        """
        Extrae parámetros de normalización de los scalers
        
        Args:
            scalers: Lista de scalers o scaler único
            param_type: 'mean' o 'scale'
        
        Returns:
            Tensor con los parámetros
        """
        params = []
        if isinstance(scalers, list):
            for scaler in scalers:
                if param_type == 'mean':
                    params.append(torch.tensor(scaler.mean_[0], dtype=torch.float32))
                else:  # scale
                    params.append(torch.tensor(scaler.scale_[0], dtype=torch.float32))
        else:
            if param_type == 'mean':
                params.append(torch.tensor(scalers.mean_[0], dtype=torch.float32))
            else:  # scale
                params.append(torch.tensor(scalers.scale_[0], dtype=torch.float32))
        
        return torch.stack(params)
        
    def forward(self, x, well_indices=None):
        """
        Forward pass con soporte para índices de pozo
        
        Args:
            x: Datos sísmicos normalizados
            well_indices: Índices de pozo para aplicaciones específicas
        """
        # Siempre pasamos well_indices al modelo base para interfaz consistente
        return self.model(x, well_indices)
    
    def desnormalize_impedance(self, impedance_norm, well_idx=0):
        """
        Desnormaliza impedancia en batch usando los parámetros del scaler específico
        
        Args:
            impedance_norm: Impedancia normalizada (tensor)
            well_idx: Índice o tensor de índices del pozo
        
        Returns:
            Impedancia física (tensor, retiene gradientes)
        """
        # Si well_idx es tensor, usarlo directamente, sino convertirlo a tensor
        if not isinstance(well_idx, torch.Tensor):
            well_idx = torch.tensor([well_idx], device=impedance_norm.device)
            
        # Obtener parámetros para los pozos especificados
        if len(well_idx.shape) == 0:  # Escalar
            impedance_mean = self.impedance_means[well_idx].to(impedance_norm.device)
            impedance_std = self.impedance_stds[well_idx].to(impedance_norm.device)
        else:  # Batch
            impedance_mean = self.impedance_means[well_idx].to(impedance_norm.device)
            impedance_std = self.impedance_stds[well_idx].to(impedance_norm.device)
            # Expandir dimensiones para broadcast
            if len(impedance_mean.shape) == 1:
                impedance_mean = impedance_mean.unsqueeze(1)
                impedance_std = impedance_std.unsqueeze(1)
        
        # Desnormalizar: x_original = x_norm * std + mean
        return impedance_norm * impedance_std + impedance_mean
    
    def normalize_seismic(self, seismic_physical, well_idx=0):
        """
        Normaliza datos sísmicos en batch usando los parámetros del scaler específico
        
        Args:
            seismic_physical: Sísmica física (tensor)
            well_idx: Índice o tensor de índices del pozo
        
        Returns:
            Sísmica normalizada (tensor, retiene gradientes)
        """
        # Si well_idx es tensor, usarlo directamente, sino convertirlo a tensor
        if not isinstance(well_idx, torch.Tensor):
            well_idx = torch.tensor([well_idx], device=seismic_physical.device)
            
        # Obtener parámetros para los pozos especificados
        if len(well_idx.shape) == 0:  # Escalar
            seismic_mean = self.seismic_means[well_idx].to(seismic_physical.device)
            seismic_std = self.seismic_stds[well_idx].to(seismic_physical.device)
        else:  # Batch
            seismic_mean = self.seismic_means[well_idx].to(seismic_physical.device)
            seismic_std = self.seismic_stds[well_idx].to(seismic_physical.device)
            # Expandir dimensiones para broadcast
            if len(seismic_mean.shape) == 1:
                seismic_mean = seismic_mean.unsqueeze(1)
                seismic_std = seismic_std.unsqueeze(1)
        
        # Normalizar: x_norm = (x_original - mean) / std
        return (seismic_physical - seismic_mean) / seismic_std
    
    def detect_zones_batch(self, seismic_data, window_size=10, variance_threshold=0.1):
        """
        Detecta zonas homogéneas en batch para pérdida física adaptativa
        
        Args:
            seismic_data: Datos sísmicos [batch_size, seq_len]
            window_size: Tamaño de ventana para análisis
            variance_threshold: Umbral para considerar zona homogénea
            
        Returns:
            homogeneous_mask: Máscara booleana de zonas homogéneas
        """
        batch_size, seq_len = seismic_data.shape
        
        if seq_len <= window_size:
            return torch.zeros_like(seismic_data, dtype=torch.bool)
        
        # Calcular varianza local usando unfold
        unfolded = seismic_data.unfold(1, window_size, 1)
        local_variance = unfolded.var(dim=2)
        
        # Pad para mantener dimensiones
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        padded_variance = F.pad(local_variance, (pad_left, pad_right), mode='replicate')
        
        # Normalizar varianza
        max_variance = padded_variance.max(dim=1, keepdim=True)[0]
        normalized_variance = padded_variance / (max_variance + 1e-8)
        
        # Crear máscara
        homogeneous_mask = normalized_variance < variance_threshold
        
        return homogeneous_mask
        
    def physics_loss(self, impedance_pred, seismic_data, wavelet, well_indices):
        """
        Cálculo de pérdida física vectorizado con adaptación por zonas
        
        Args:
            impedance_pred: Impedancia predicha (normalizada) [batch_size, seq_len]
            seismic_data: Datos sísmicos normalizados [batch_size, seq_len]
            wavelet: Wavelet para convolución [wavelet_length]
            well_indices: Tensor con índices de pozo para cada muestra [batch_size]
            
        Returns:
            Pérdida física promedio para el batch
        """
        batch_size = impedance_pred.shape[0]
        
        # Verificar que la impedancia predicha tenga al menos 1 dimensión
        if len(impedance_pred.shape) == 1:
            impedance_pred = impedance_pred.unsqueeze(1)
        
        seq_len = impedance_pred.shape[1]
        
        # Verificar dimensiones
        if seismic_data.shape[1] != seq_len:
            print(f"Dimensiones no coincidentes: seismic={seismic_data.shape}, impedance={impedance_pred.shape}")
            return self.mse(impedance_pred, impedance_pred.detach()) * 0.1
        
        # Para secuencias muy cortas, usar pérdida simple
        if seq_len == 1:
            return self.mse(impedance_pred, seismic_data)
        
        # Detectar zonas homogéneas en los datos sísmicos
        homogeneous_zones = self.detect_zones_batch(seismic_data)
        
        # Convertir wavelet a tensor
        if not isinstance(wavelet, torch.Tensor):
            wavelet_tensor = torch.tensor(wavelet, dtype=torch.float32, device=impedance_pred.device)
        else:
            wavelet_tensor = wavelet.to(impedance_pred.device)
        
        # Preparar wavelet para convolución
        wavelet_conv = wavelet_tensor.view(1, 1, -1)
        wavelet_flipped = torch.flip(wavelet_conv, [2])
        
        # Procesar en mini-batches
        mini_batch_size = min(16, batch_size)
        batch_loss = 0.0
        zone_penalty = 0.0
        
        try:
            for i in range(0, batch_size, mini_batch_size):
                end_idx = min(i + mini_batch_size, batch_size)
                current_batch_size = end_idx - i
                
                # Extraer mini-batch
                batch_impedance = impedance_pred[i:end_idx]
                batch_seismic = seismic_data[i:end_idx]
                batch_well_indices = well_indices[i:end_idx]
                batch_zones = homogeneous_zones[i:end_idx]
                
                # 1. Desnormalizar impedancia
                impedance_physical = self.desnormalize_impedance(batch_impedance, batch_well_indices)
                
                # 2. Calcular coeficientes de reflexión
                reflection_coeff = torch.zeros_like(impedance_physical)
                reflection_coeff[:, 1:] = (impedance_physical[:, 1:] - impedance_physical[:, :-1]) / \
                                       (impedance_physical[:, 1:] + impedance_physical[:, :-1] + 1e-10)
                
                # 3. Convolución con wavelet
                refl_coeff_reshaped = reflection_coeff.view(current_batch_size, 1, seq_len)
                padding = len(wavelet_tensor) // 2
                
                seismic_pred = F.conv1d(refl_coeff_reshaped, wavelet_flipped, padding=padding)
                
                # Asegurar dimensiones correctas
                if seismic_pred.shape[-1] != seq_len:
                    seismic_pred = F.interpolate(seismic_pred, size=seq_len, mode='linear', align_corners=False)
                
                seismic_pred = seismic_pred.view(current_batch_size, seq_len)
                
                # 4. Normalizar la sísmica predicha
                seismic_pred_norm = self.normalize_seismic(seismic_pred, batch_well_indices)
                
                # 5. Calcular error básico
                basic_error = (seismic_pred_norm - batch_seismic) ** 2
                
                # 6. Aplicar penalización adicional en zonas homogéneas
                # En zonas homogéneas, penalizar más fuertemente las diferencias
                zone_weights = torch.ones_like(basic_error)
                zone_weights[batch_zones] = 2.0  # Doble peso en zonas homogéneas
                
                weighted_error = basic_error * zone_weights
                
                # 7. Calcular pérdida para este mini-batch
                mini_batch_loss = torch.mean(weighted_error)
                batch_loss += mini_batch_loss * current_batch_size
                
                # 8. Penalización adicional por variación en zonas homogéneas
                if torch.any(batch_zones):
                    # Calcular variación de impedancia en zonas homogéneas
                    impedance_diff = torch.abs(impedance_physical[:, 1:] - impedance_physical[:, :-1])
                    zone_mask = batch_zones[:, 1:]  # Ajustar para diferencias
                    
                    if torch.any(zone_mask):
                        zone_variation = impedance_diff * zone_mask.float()
                        zone_penalty += torch.mean(zone_variation) * 0.1
                
        except Exception as e:
            print(f"Error en convolución física: {e}")
            return self.mse(impedance_pred, impedance_pred.detach()) * 0.1
        
        # Pérdida total
        total_loss = (batch_loss / batch_size) + zone_penalty
        
        return total_loss

# Dataset mejorado con mejor distribución de ventanas y procesamiento optimizado
class ImprovedSeismicDataset(torch.utils.data.Dataset):
    def __init__(self, seismic_data, impedance_data, window_size=64, use_weighting=True, 
                 overlap_factor=6, well_indices=None, device=None):
        """
        Dataset optimizado con procesamiento por lotes
        
        Args:
            seismic_data: Lista de trazas sísmicas normalizadas
            impedance_data: Lista de impedancias acústicas normalizadas
            window_size: Tamaño de la ventana para el contexto
            use_weighting: Si se debe aplicar ponderación a la ventana
            overlap_factor: Factor de solapamiento (mayor número = más solapamiento)
            well_indices: Lista para rastrear a qué pozo pertenece cada ventana
            device: Dispositivo para los tensores (None = mantener en CPU para no agotar memoria GPU)
        """
        super().__init__()
        self.window_size = window_size
        self.device = device  # Guardamos pero no lo usamos para la inicialización
        
        # Generar ventana Hann una sola vez (evitar recalcular)
        if use_weighting:
            window_weights = 0.5 * (1 - torch.cos(2 * torch.pi * torch.arange(window_size) / (window_size - 1)))
            self.window_weights = window_weights
        else:
            self.window_weights = torch.ones(window_size)
            
        # Preparar listas para datos
        seismic_windows = []
        impedance_targets = []
        well_indices_list = []
        
        # Stride optimizado para solapamiento
        stride = max(1, window_size // overlap_factor)
        
        # Procesar todas las trazas
        for i, (seismic, impedance) in enumerate(zip(seismic_data, impedance_data)):
            seismic_tensor = torch.tensor(seismic, dtype=torch.float32)
            impedance_tensor = torch.tensor(impedance, dtype=torch.float32)
            
            # Crear ventanas deslizantes con alto solapamiento
            for j in range(0, len(seismic) - window_size + 1, stride):
                # Extraer ventana
                window = seismic_tensor[j:j+window_size]
                
                # Aplicar ponderación si se solicita
                if use_weighting:
                    window = window * self.window_weights
                
                seismic_windows.append(window)
                
                # El target es el valor central de impedancia para enfoque local
                center_idx = j + window_size//2
                impedance_targets.append(impedance_tensor[center_idx])
                
                # Guardar el índice del pozo
                well_indices_list.append(i)
        
        # Convertir a tensores pero mantenerlos en CPU
        self.seismic_windows = torch.stack(seismic_windows)
        self.impedance_targets = torch.tensor(impedance_targets).unsqueeze(1)
        self.well_indices = torch.tensor(well_indices_list, dtype=torch.long)
        
    def __len__(self):
        return len(self.seismic_windows)
    
    def __getitem__(self, idx):
        # Transferir a GPU solo cuando se accede al elemento individual
        # No es necesario transferir a GPU aquí si el DataLoader tiene pin_memory=True
        return self.seismic_windows[idx], self.impedance_targets[idx], self.well_indices[idx]

# Early Stopping optimizado
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, verbose=True):
        """Early stopping mejorado con mayor robustez"""
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = min_delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        '''Guarda el modelo cuando la validación mejora'''
        if self.verbose:
            print(f'   Pérdida de validación disminuyó ({self.val_loss_min:.6f} --> {val_loss:.6f}). Guardando modelo...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
