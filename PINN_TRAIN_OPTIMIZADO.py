"""
PINN_TRAIN_OPTIMIZADO.py - Funciones de Entrenamiento Optimizadas
PARTE 1: Función principal de entrenamiento por fold

Características:
- Entrenamiento acelerado con AMP (precisión mixta automática)
- Optimizadores mejorados y learning rate schedulers
- Hooks de progreso y monitoreo eficientes
- MEJORADO: Uso de pérdida adaptativa por zonas
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Dispositivo global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fold_optimized(fold, train_ids, val_ids, data_bundle, config):
    """
    Entrena un modelo para un fold específico con optimización GPU y pérdida adaptativa
    
    Args:
        fold: Número de fold actual
        train_ids: Índices de muestras de entrenamiento
        val_ids: Índices de muestras de validación
        data_bundle: Bundle de datos
        config: Diccionario de configuración
        
    Returns:
        Modelo entrenado y resultados
    """
    from collections import defaultdict
    
    print(f"\n=== ENTRENANDO FOLD {fold+1}/{config['k_folds']} ===\n")
    
    # Habilitar cuDNN benchmark para acelerar convoluciones
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Importar modelos necesarios
    from PINN_1B_OPTIMIZADO import MultiScaleImpedanceModel, EnhancedSpikeControlLoss
    from PINN_1C_OPTIMIZADO import NormalizedPhysicsWrapper, EarlyStopping
    
    train_dataset = data_bundle['train_dataset']
    
    # Configurar loaders para este fold
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    
    # Asegurar batch_size mínimo para BatchNorm
    batch_size = max(4, config['batch_size'])
    
    fold_train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_subsampler,
        num_workers=0,  # Reducir a 0 para evitar problemas de CUDA
        pin_memory=True,
        persistent_workers=False
    )
    
    fold_val_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=val_subsampler,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    
    # Inicializar modelo
    base_model = MultiScaleImpedanceModel(
        input_size=data_bundle['window_size'], 
        hidden_size=config['hidden_size'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Envolver con wrapper físico
    model = NormalizedPhysicsWrapper(
        base_model,
        data_bundle['seismic_scalers'],
        data_bundle['impedance_scalers']
    ).to(device)
    
    # Optimizador AdamW
    optimizer = optim.AdamW(
        base_model.parameters(),
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # MODIFICACIÓN AQUÍ: Usar la nueva función de pérdida mejorada
    criterion = EnhancedSpikeControlLoss(config)
    use_adaptive_loss = False
    print("Usando pérdida mejorada para control de spikes")
    
    # Scheduler OneCycleLR
    steps_per_epoch = len(fold_train_loader)
    total_steps = steps_per_epoch * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 3,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        verbose=True
    )
    
    model_path = os.path.join(config['results_dir'], f'model_fold_{fold+1}_best.pth')
    
    # Para seguimiento del entrenamiento
    metrics = defaultdict(list)
    
    # Convertir wavelet a tensor
    wavelet_tensor = torch.tensor(data_bundle['wavelet'], dtype=torch.float32).to(device)
    
    # Inicializar scaler para precisión mixta
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Entrenamiento
    base_model.train()
    fold_start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Calcular peso físico actual
        if epoch < config['epochs_to_increase']:
            progress = (epoch / config['epochs_to_increase'])**2
            current_physics_weight = config['initial_physics_weight'] + \
                (config['final_physics_weight'] - config['initial_physics_weight']) * progress
        else:
            current_physics_weight = config['final_physics_weight']
        
        # Barra de progreso
        progress_bar = tqdm(fold_train_loader, desc=f"Época {epoch+1}/{config['epochs']}")
        
        for batch_idx, (batch_seismic, batch_impedance, well_indices) in enumerate(progress_bar):
            # Saltar batches pequeños
            if batch_seismic.size(0) < 2:
                continue
                
            # Transferir a GPU
            batch_seismic = batch_seismic.to(device, non_blocking=True)
            batch_impedance = batch_impedance.to(device, non_blocking=True)
            well_indices = well_indices.to(device, non_blocking=True)
            
            # Verificar dimensiones en el primer batch
            if batch_idx == 0 and epoch == 0:
                print(f"  Dimensiones - seismic: {batch_seismic.shape}, impedance: {batch_impedance.shape}")
            
            # Limpiar gradientes
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass con precisión mixta
            with autocast(enabled=torch.cuda.is_available()):
                # Forward pass
                pred_impedance = model(batch_seismic, well_indices)
                
                # Asegurar forma correcta
                if len(pred_impedance.shape) == 1:
                    pred_impedance = pred_impedance.unsqueeze(1)
                
                # Calcular pérdida de datos con la nueva función
                data_loss = criterion(pred_impedance, batch_impedance)
                
                # Preparar entrada para pérdida física
                if batch_seismic.shape[1] != pred_impedance.shape[1]:
                    center_idx = batch_seismic.shape[1] // 2
                    if pred_impedance.shape[1] == 1:
                        batch_seismic_center = batch_seismic[:, center_idx].unsqueeze(1)
                        physics_input_seismic = batch_seismic_center
                    else:
                        physics_input_seismic = batch_seismic
                else:
                    physics_input_seismic = batch_seismic
                
                # Pérdida física
                try:
                    physics_error = model.physics_loss(
                        pred_impedance, 
                        physics_input_seismic,
                        wavelet_tensor,
                        well_indices
                    )
                except Exception as e:
                    print(f"Error en pérdida física: {e}")
                    physics_error = torch.tensor(0.1, device=device, requires_grad=True)
                
                # Pérdida combinada
                loss = data_loss + current_physics_weight * physics_error
            
            # Backward y optimización
            scaler.scale(loss).backward()
            
            # Clip gradientes
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=3.0)
            
            # Actualizar pesos
            scaler.step(optimizer)
            scaler.update()
            
            # Actualizar learning rate
            scheduler.step()
            
            # Acumular métricas
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['data_loss'] += data_loss.item()
            epoch_metrics['physics_loss'] += physics_error.item()
            num_batches += 1
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'data': f"{data_loss.item():.4f}",
                'phys': f"{physics_error.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Promediar métricas para esta época
        if num_batches > 0:
            for k in epoch_metrics:
                epoch_metrics[k] /= num_batches
        
        # Guardar métricas
        metrics['train_loss'].append(epoch_metrics['loss'])
        metrics['train_data_loss'].append(epoch_metrics['data_loss'])
        metrics['train_physics_loss'].append(epoch_metrics['physics_loss'])
        
        # Información de época
        epoch_time = time.time() - epoch_start
        print(f'Época {epoch+1}/{config["epochs"]} ({epoch_time:.2f}s) - '
              f'Loss: {epoch_metrics["loss"]:.6f}, '
              f'Data: {epoch_metrics["data_loss"]:.6f}, '
              f'Physics: {epoch_metrics["physics_loss"]:.6f}, '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')
        
        # Evaluación en validación
        val_metrics = evaluate_model(model, fold_val_loader, criterion, 
                                    wavelet_tensor, current_physics_weight, use_adaptive_loss)
        
        # Guardar métricas de validación
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_data_loss'].append(val_metrics['data_loss'])
        metrics['val_physics_loss'].append(val_metrics['physics_loss'])
        
        print(f'  Validación - Loss: {val_metrics["loss"]:.6f}, '
              f'Data: {val_metrics["data_loss"]:.6f}, '
              f'Physics: {val_metrics["physics_loss"]:.6f}')
        
        # Gráfico de progreso
        plot_path = os.path.join(config['results_dir'], f'learning_curves_fold_{fold+1}.png')
        plot_learning_curves_optimized(
            metrics['train_loss'], metrics['val_loss'],
            metrics['train_data_loss'], metrics['val_data_loss'],
            metrics['train_physics_loss'], metrics['val_physics_loss'],
            epoch, fold, save_path=plot_path
        )
        
        # Early stopping
        early_stopping(val_metrics['loss'], base_model, model_path)
        if early_stopping.early_stop:
            print(f"  Early stopping en época {epoch+1}")
            break
        
        # Volver a modo entrenamiento
        base_model.train()
    
    # Tiempo total del fold
    fold_time = time.time() - fold_start_time
    print(f'\nFold {fold+1} completado en {fold_time/60:.2f} minutos')
    
    # Cargar mejor modelo
    best_base_model = MultiScaleImpedanceModel(
        input_size=data_bundle['window_size'], 
        hidden_size=config['hidden_size'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    best_base_model.load_state_dict(torch.load(model_path))
    best_base_model.eval()
    
    # Envolver modelo para uso
    best_model = NormalizedPhysicsWrapper(
        best_base_model,
        data_bundle['seismic_scalers'],
        data_bundle['impedance_scalers']
    ).to(device)
    
    # Resultados
    fold_results = {
        'train_losses': metrics['train_loss'],
        'val_losses': metrics['val_loss'],
        'data_losses': metrics['train_data_loss'],
        'physics_losses': metrics['train_physics_loss'],
        'val_data_losses': metrics['val_data_loss'],
        'val_physics_losses': metrics['val_physics_loss'],
        'best_val_loss': early_stopping.val_loss_min,
        'epochs_trained': len(metrics['val_loss'])
    }
    
    return best_model, fold_results
    
@torch.no_grad()
def evaluate_model(model, dataloader, criterion, wavelet_tensor, physics_weight, use_adaptive_loss=False):
    """
    Evalúa el modelo en el conjunto de validación con optimizaciones
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de validación
        criterion: Función de pérdida para datos
        wavelet_tensor: Wavelet para cálculos físicos
        physics_weight: Peso para la pérdida física
        use_adaptive_loss: Si se usa pérdida adaptativa combinada
        
    Returns:
        Diccionario con métricas de evaluación
    """
    model.eval()
    metrics = {'loss': 0.0, 'data_loss': 0.0, 'physics_loss': 0.0}
    num_batches = 0
    
    # Usar precisión mixta para evaluación también
    with autocast(enabled=torch.cuda.is_available()):
        for batch_seismic, batch_impedance, well_indices in dataloader:
            if batch_seismic.size(0) < 2:
                continue
                
            # Transferir a GPU
            batch_seismic = batch_seismic.to(device, non_blocking=True)
            batch_impedance = batch_impedance.to(device, non_blocking=True)
            well_indices = well_indices.to(device, non_blocking=True)
            
            # Forward pass
            pred_impedance = model(batch_seismic, well_indices)
            
            # Asegurar la forma correcta
            if len(pred_impedance.shape) == 1:
                pred_impedance = pred_impedance.unsqueeze(1)
            
            # Pérdida de datos - MODIFICACIÓN AQUÍ
            data_loss = criterion(pred_impedance, batch_impedance)
            
            # Preparar entrada para pérdida física
            if batch_seismic.shape[1] != pred_impedance.shape[1]:
                center_idx = batch_seismic.shape[1] // 2
                if pred_impedance.shape[1] == 1:
                    batch_seismic_center = batch_seismic[:, center_idx].unsqueeze(1)
                    physics_input_seismic = batch_seismic_center
                else:
                    physics_input_seismic = batch_seismic
            else:
                physics_input_seismic = batch_seismic
            
            # Pérdida física
            try:
                physics_error = model.physics_loss(
                    pred_impedance, 
                    physics_input_seismic, 
                    wavelet_tensor,
                    well_indices
                )
            except Exception as e:
                print(f"Error en pérdida física durante evaluación: {e}")
                physics_error = torch.tensor(0.1, device=device)
            
            # Pérdida total
            loss = data_loss + physics_weight * physics_error
            
            # Acumular métricas
            metrics['loss'] += loss.item()
            metrics['data_loss'] += data_loss.item()
            metrics['physics_loss'] += physics_error.item()
            num_batches += 1
    
    # Promediar métricas
    if num_batches > 0:
        for k in metrics:
            metrics[k] /= num_batches
    
    return metrics

def plot_learning_curves_optimized(train_losses, val_losses, data_losses, val_data_losses,
                                  physics_losses, val_physics_losses, epoch, fold, save_path=None):
    """
    Visualiza curvas de aprendizaje optimizadas
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación
        data_losses: Lista de pérdidas de datos en entrenamiento
        val_data_losses: Lista de pérdidas de datos en validación
        physics_losses: Lista de pérdidas físicas en entrenamiento
        val_physics_losses: Lista de pérdidas físicas en validación
        epoch: Época actual
        fold: Fold actual
        save_path: Ruta para guardar el gráfico
    """
    # Usar el backend no interactivo para ahorrar memoria
    plt.switch_backend('agg')
    
    plt.figure(figsize=(15, 12), dpi=100)
    
    # Pérdidas totales
    plt.subplot(3, 1, 1)
    plt.title(f'Curvas de Aprendizaje - Fold {fold+1}, Época {epoch+1}')
    plt.plot(train_losses, 'b-', label='Pérdida Total Entrenamiento')
    plt.plot(val_losses, 'r-', label='Pérdida Total Validación')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.legend()
    
    # Pérdidas de datos
    plt.subplot(3, 1, 2)
    plt.plot(data_losses, 'g-', label='Pérdida Datos (Train)')
    plt.plot(val_data_losses, 'g--', label='Pérdida Datos (Val)')
    plt.ylabel('Pérdida Datos')
    plt.grid(True)
    plt.legend()
    
    # Pérdidas físicas
    plt.subplot(3, 1, 3)
    plt.plot(physics_losses, 'm-', label='Pérdida Física (Train)')
    plt.plot(val_physics_losses, 'm--', label='Pérdida Física (Val)')
    plt.ylabel('Pérdida Física')
    plt.xlabel('Época')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close()

def create_ensemble_optimized(data_bundle, config):
    """
    Entrena múltiples modelos y crea un ensemble optimizado
    
    Args:
        data_bundle: Bundle de datos
        config: Diccionario de configuración
        
    Returns:
        Resultados del ensemble
    """
    print(f"\n=== CREANDO ENSEMBLE DE {config['num_models_ensemble']} MODELOS ===\n")
    
    # Preparar para validación cruzada
    kf = data_bundle['kfold']
    X = data_bundle['X_train']
    
    # Resultados
    all_models = []
    all_results = []
    
    # Entrenar cada fold
    for fold, (train_ids, val_ids) in enumerate(kf.split(X)):
        if fold >= config['num_models_ensemble']:
            break
            
        # Entrenar modelo para este fold
        print(f"\nEntrenando modelo {fold+1}/{config['num_models_ensemble']} en {device}")
        model, results = train_fold_optimized(fold, train_ids, val_ids, data_bundle, config)
        
        # Guardar modelo y resultados
        all_models.append(model)
        all_results.append(results)
        
        # Guardar modelo con compresión
        model_dir = os.path.join(config['results_dir'], 'ensemble_models')
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(
            model.model.state_dict(), 
            os.path.join(model_dir, f'ensemble_model_{fold+1}.pth'),
            _use_new_zipfile_serialization=True
        )
    
    # Encontrar mejor modelo por validación
    best_fold = np.argmin([r['best_val_loss'] for r in all_results])
    
    print("\nResumen de validación cruzada:")
    for i, res in enumerate(all_results):
        print(f'  Fold {i+1}: {res["best_val_loss"]:.6f} ({res["epochs_trained"]} épocas)')
    print(f'  Mejor modelo: Fold {best_fold+1} con pérdida {all_results[best_fold]["best_val_loss"]:.6f}')
    
    # Estructura de resultados
    ensemble_results = {
        'models': all_models,
        'fold_results': all_results,
        'best_fold': best_fold,
        'best_model': all_models[best_fold]
    }
    
    return ensemble_results

def prepare_data_optimized(config):
    """
    Prepara datos para entrenamiento con optimizaciones para GPU
    
    Args:
        config: Diccionario de configuración
    
    Returns:
        Bundle de datos optimizado
    """
    import glob
    import random
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from PINN_1A_OPTIMIZADO import read_las_file, generate_synthetic_seismic, add_noise, ricker_wavelet
    from PINN_1C_OPTIMIZADO import ImprovedSeismicDataset
    
    print("\n=== PREPARANDO DATOS OPTIMIZADOS ===\n")
    
    # Configuración inicial
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Generar wavelet Ricker normalizado
    wavelet = ricker_wavelet(config['freq'], config['length'], config['dt'])
    wavelet = wavelet / np.max(np.abs(wavelet))
    
    # Leer archivos LAS
    las_files = glob.glob(os.path.join(config['data_path'], config['las_pattern']))
    if not las_files:
        raise FileNotFoundError(f"No se encontraron archivos LAS en {config['data_path']}")
    
    # Selección de pozos
    if config['num_wells'] != 'all' and config['num_wells'] < len(las_files):
        las_files = random.sample(las_files, config['num_wells'])
    
    print(f"Procesando {len(las_files)} pozos...")
    
    # Procesamiento paralelo para muchos pozos
    from concurrent.futures import ThreadPoolExecutor
    
    # Leer datos de pozos
    well_data = []
    with ThreadPoolExecutor(max_workers=min(8, len(las_files))) as executor:
        results = list(executor.map(read_las_file, las_files))
        for data in results:
            if data is not None:
                well_data.append(data)
                print(f"  Leído: {data['well_name']}")
    
    # Generar datos sísmicos sintéticos
    seismic_traces = []
    noisy_seismic_traces = []
    acoustic_impedances = []
    
    for well in well_data:
        # Traza sísmica limpia
        seismic = generate_synthetic_seismic(well['acoustic_impedance'], wavelet)
        # Traza con ruido
        noisy_seismic = add_noise(seismic, config['snr_db'])
        
        seismic_traces.append(seismic)
        noisy_seismic_traces.append(noisy_seismic)
        acoustic_impedances.append(well['acoustic_impedance'])
    
    # Normalización por pozo individual
    seismic_scalers = []
    impedance_scalers = []
    normalized_seismic = []
    normalized_impedance = []
    
    for i, (seis, imp) in enumerate(zip(noisy_seismic_traces, acoustic_impedances)):
        # Normalizar sísmica
        s_scaler = StandardScaler()
        norm_seis = s_scaler.fit_transform(seis.reshape(-1, 1)).flatten()
        
        # Normalizar impedancia
        i_scaler = StandardScaler()
        norm_imp = i_scaler.fit_transform(imp.reshape(-1, 1)).flatten()
        
        seismic_scalers.append(s_scaler)
        impedance_scalers.append(i_scaler)
        normalized_seismic.append(norm_seis)
        normalized_impedance.append(norm_imp)
    
    # Preparar dataset - mantener datos en CPU
    train_dataset = ImprovedSeismicDataset(
        normalized_seismic,
        normalized_impedance,
        config['window_size'],
        config['use_weighting'],
        overlap_factor=config['stride_factor'],
        well_indices=list(range(len(well_data)))
    )
    
    # Configurar KFold para validación cruzada
    kf = KFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
    X_train = np.arange(len(train_dataset))  # Índices para kfold
    
    # Estructura de datos para entrenamiento
    original_data = {
        'well_data': well_data,
        'seismic_traces': seismic_traces,
        'noisy_seismic_traces': noisy_seismic_traces,
        'acoustic_impedances': acoustic_impedances,
        'normalized_seismic': normalized_seismic,
        'normalized_impedance': normalized_impedance,
        'test_indices': list(range(len(well_data)))
    }
    
    # Crear window_weights
    window_weights = None
    if config['use_weighting']:
        window_weights = torch.tensor(hann_window(config['window_size']))
    
    data_bundle = {
        'train_dataset': train_dataset,
        'seismic_scalers': seismic_scalers,
        'impedance_scalers': impedance_scalers,
        'wavelet': wavelet,
        'window_size': config['window_size'],
        'window_weights': window_weights,
        'kfold': kf,
        'X_train': X_train,
        'original_data': original_data
    }
    
    print(f"\nDatos preparados: {len(train_dataset)} muestras de entrenamiento")
    
    # Análisis de calidad de datos
    print("\nAnálisis de calidad de datos:")
    for i, well in enumerate(well_data[:3]):  # Primeros 3 pozos
        impedance = well['acoustic_impedance']
        print(f"  {well['well_name']}:")
        print(f"    Rango: [{np.min(impedance):.1f}, {np.max(impedance):.1f}]")
        print(f"    Media: {np.mean(impedance):.1f}, Std: {np.std(impedance):.1f}")
    
    return data_bundle

def hann_window(size):
    """Crea una ventana Hann para ponderar muestras"""
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
    


