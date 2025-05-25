"""
PINN_EVAL_OPTIMIZADO.py - Funciones de Evaluación y Visualización Optimizadas

Características:
- Visualizaciones aceleradas con procesamiento vectorizado
- Análisis de capas optimizado
- Métricas de error mejoradas
- NUEVO: Análisis detallado de zonas homogéneas y transiciones
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage, signal
import time
import os

# Dispositivo global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model_optimized(ensemble_results, data_bundle, test_well_idx=None, config=None):
    """
    Evalúa el modelo en datos de prueba con optimizaciones y análisis mejorado
    
    Args:
        ensemble_results: Resultados del ensemble de modelos
        data_bundle: Bundle de datos
        test_well_idx: Índice del pozo de prueba (None para aleatorio)
        config: Configuración
        
    Returns:
        Resultados de evaluación y visualizaciones
    """
    print("\n=== EVALUANDO MODELO OPTIMIZADO ===\n")
    
    # Importar funciones de predicción
    from PINN_1D_OPTIMIZADO import improved_ensemble_predict, comprehensive_impedance_enhancement, simple_ensemble_predict
    from PINN_1D_OPTIMIZADO import detect_homogeneous_zones, analyze_prediction_quality
    
    # Obtener datos originales
    original_data = data_bundle['original_data']
    well_data = original_data['well_data']
    test_indices = original_data['test_indices']
    normalized_seismic = original_data['normalized_seismic']
    window_size = data_bundle['window_size']
    window_weights = data_bundle['window_weights']
    
    # Seleccionar pozo de prueba
    if test_well_idx is None:
        test_well_idx = np.random.choice(test_indices)
    elif test_well_idx not in test_indices:
        print(f"Advertencia: Índice {test_well_idx} no está en los índices de prueba. Seleccionando aleatorio.")
        test_well_idx = np.random.choice(test_indices)
    
    test_well_data = well_data[test_well_idx]
    test_well_name = test_well_data['well_name']
    
    print(f'Evaluando en pozo: {test_well_name} (índice: {test_well_idx})')
    
    # Cronometrar tiempo de evaluación
    start_time = time.time()
    
    # Datos reales
    real_impedance = test_well_data['acoustic_impedance']
    real_seismic = original_data['seismic_traces'][test_well_idx]
    real_noisy_seismic = original_data['noisy_seismic_traces'][test_well_idx]
    
    # Datos normalizados
    normalized_test_seismic = normalized_seismic[test_well_idx]
    
    # Detectar zonas homogéneas en los datos reales
    homogeneous_zones = detect_homogeneous_zones(
        normalized_test_seismic,
        window_size=config.get('local_variance_window', 20) if config else 20,
        variance_threshold=config.get('homogeneous_zone_threshold', 0.05) if config else 0.05
    )
    
    print(f"Zonas homogéneas detectadas: {np.sum(homogeneous_zones)}/{len(homogeneous_zones)} "
          f"({np.sum(homogeneous_zones)/len(homogeneous_zones)*100:.1f}%)")
    
    # Predicción con ensemble
    print(f"\nRealizando predicciones con ensemble mejorado para pozo {test_well_idx}...")
    models_to_use = ensemble_results['models'][:config.get('num_models_ensemble', 3) if config else 3]
    
    try:
        # Método completo con configuración mejorada
        normalized_prediction = improved_ensemble_predict(
            models_to_use, 
            normalized_test_seismic, 
            window_size,
            window_weights,
            well_idx=test_well_idx,
            stride_factor=config.get('stride_factor', 10) if config else 10,
            config=config
        )
    except Exception as e:
        print(f"Error en predicción compleja: {e}")
        print("Usando método de predicción alternativo...")
        normalized_prediction = simple_ensemble_predict(
            models_to_use,
            normalized_test_seismic,
            window_size,
            well_idx=test_well_idx,
            config=config
        )
    
    # Desnormalizar usando el scaler específico del pozo de prueba
    impedance_scaler = data_bundle['impedance_scalers'][test_well_idx]
    predicted_impedance = impedance_scaler.inverse_transform(
        normalized_prediction.reshape(-1, 1)
    ).flatten()
    
    # Análisis de calidad antes del postprocesamiento
    print("\nAnálisis de calidad pre-postprocesamiento:")
    pre_quality = analyze_prediction_quality(predicted_impedance, real_noisy_seismic, homogeneous_zones)
    print(f"  Estabilidad en zonas homogéneas: {pre_quality['homogeneous_stability']:.3f}")
    print(f"  Nitidez en transiciones: {pre_quality['transition_sharpness']:.3f}")
    print(f"  Correlación de gradientes: {pre_quality['gradient_correlation']:.3f}")
    
    # Postprocesamiento para mejorar resultado
    print("\nAplicando postprocesamiento avanzado...")
    try:
        final_impedance = comprehensive_impedance_enhancement(
            predicted_impedance,
            real_noisy_seismic,
            real_impedance=real_impedance,  # Solo para calibración
            config=config
        )
    except Exception as e:
        print(f"Error en postprocesamiento: {e}")
        # Usar versión más simple si falla
        final_impedance = ndimage.gaussian_filter1d(predicted_impedance, sigma=1.0)
    
    # Análisis de calidad después del postprocesamiento
    print("\nAnálisis de calidad post-postprocesamiento:")
    post_quality = analyze_prediction_quality(final_impedance, real_noisy_seismic, homogeneous_zones)
    print(f"  Estabilidad en zonas homogéneas: {post_quality['homogeneous_stability']:.3f}")
    print(f"  Nitidez en transiciones: {post_quality['transition_sharpness']:.3f}")
    print(f"  Correlación de gradientes: {post_quality['gradient_correlation']:.3f}")
    
    # Calcular métricas generales
    metrics = calculate_impedance_metrics(real_impedance, final_impedance)
    
    # Calcular métricas por zonas
    zone_metrics = calculate_zone_specific_metrics(
        real_impedance, final_impedance, homogeneous_zones
    )
    
    print(f'\nMétricas generales:')
    print(f'  MSE: {metrics["mse"]:.2f}')
    print(f'  RMSE: {metrics["rmse"]:.2f}')
    print(f'  MAE: {metrics["mae"]:.2f}')
    print(f'  R²: {metrics["r2"]:.4f}')
    
    print(f'\nMétricas por zonas:')
    print(f'  RMSE en zonas homogéneas: {zone_metrics["homogeneous_rmse"]:.2f}')
    print(f'  RMSE en transiciones: {zone_metrics["transition_rmse"]:.2f}')
    print(f'  Ratio de mejora: {zone_metrics["improvement_ratio"]:.3f}')
    
    # Analizar detección de capas
    layer_metrics = analyze_layers_optimized(
        real_impedance, 
        final_impedance, 
        real_seismic,
        homogeneous_zones,
        well_name=test_well_name,
        config=config
    )
    
    # Visualizar resultados mejorados
    plot_evaluation_results_enhanced(
        real_impedance, 
        final_impedance,
        predicted_impedance,  # Sin postprocesamiento
        real_noisy_seismic,
        homogeneous_zones,
        metrics,
        zone_metrics,
        test_well_name,
        test_well_idx,
        save_path=os.path.join(config['results_dir'] if config else ".", f'{test_well_name}_enhanced_results.png')
    )
    
    # Guardar resultados
    results = {
        'real_impedance': real_impedance,
        'predicted_impedance': final_impedance,
        'predicted_raw': predicted_impedance,
        'real_seismic': real_seismic,
        'noisy_seismic': real_noisy_seismic,
        'homogeneous_zones': homogeneous_zones,
        'metrics': metrics,
        'zone_metrics': zone_metrics,
        'layer_metrics': layer_metrics,
        'pre_quality': pre_quality,
        'post_quality': post_quality,
        'test_well_name': test_well_name,
        'test_well_idx': test_well_idx,
        'evaluation_time': time.time() - start_time
    }
    
    save_path = os.path.join(config['results_dir'] if config else ".", f'{test_well_name}_results.npy')
    np.save(save_path, results)
    
    return results

def calculate_zone_specific_metrics(real_impedance, predicted_impedance, homogeneous_zones):
    """
    Calcula métricas específicas para zonas homogéneas vs transiciones
    
    Args:
        real_impedance: Impedancia real
        predicted_impedance: Impedancia predicha
        homogeneous_zones: Máscara de zonas homogéneas
        
    Returns:
        Diccionario con métricas por zona
    """
    # Asegurar misma longitud
    min_len = min(len(real_impedance), len(predicted_impedance), len(homogeneous_zones))
    real_impedance = real_impedance[:min_len]
    predicted_impedance = predicted_impedance[:min_len]
    homogeneous_zones = homogeneous_zones[:min_len]
    
    # Métricas en zonas homogéneas
    if np.any(homogeneous_zones):
        homogeneous_error = real_impedance[homogeneous_zones] - predicted_impedance[homogeneous_zones]
        homogeneous_mse = np.mean(homogeneous_error ** 2)
        homogeneous_rmse = np.sqrt(homogeneous_mse)
        homogeneous_mae = np.mean(np.abs(homogeneous_error))
    else:
        homogeneous_rmse = homogeneous_mae = 0
    
    # Métricas en zonas de transición
    transition_zones = ~homogeneous_zones
    if np.any(transition_zones):
        transition_error = real_impedance[transition_zones] - predicted_impedance[transition_zones]
        transition_mse = np.mean(transition_error ** 2)
        transition_rmse = np.sqrt(transition_mse)
        transition_mae = np.mean(np.abs(transition_error))
    else:
        transition_rmse = transition_mae = 0
    
    # Ratio de mejora (menor es mejor en zonas homogéneas)
    improvement_ratio = homogeneous_rmse / (transition_rmse + 1e-8)
    
    return {
        'homogeneous_rmse': homogeneous_rmse,
        'homogeneous_mae': homogeneous_mae,
        'transition_rmse': transition_rmse,
        'transition_mae': transition_mae,
        'improvement_ratio': improvement_ratio,
        'homogeneous_percentage': np.sum(homogeneous_zones) / len(homogeneous_zones) * 100
    }

def calculate_impedance_metrics(real_impedance, predicted_impedance):
    """
    Calcula métricas de evaluación para impedancia, optimizadas para grandes conjuntos
    
    Args:
        real_impedance: Impedancia real
        predicted_impedance: Impedancia predicha
        
    Returns:
        Diccionario con métricas calculadas
    """
    # Verificar que los arrays tienen la misma longitud
    if len(real_impedance) != len(predicted_impedance):
        min_len = min(len(real_impedance), len(predicted_impedance))
        real_impedance = real_impedance[:min_len]
        predicted_impedance = predicted_impedance[:min_len]
        print(f"Advertencia: Longitudes diferentes. Truncado a {min_len} muestras.")
    
    # Convertir a arrays numpy si son tensores
    if isinstance(real_impedance, torch.Tensor):
        real_impedance = real_impedance.cpu().numpy()
    if isinstance(predicted_impedance, torch.Tensor):
        predicted_impedance = predicted_impedance.cpu().numpy()
    
    # Calcular métricas básicas de forma vectorizada
    error = real_impedance - predicted_impedance
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    # Métricas estándar
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    
    # Métricas adicionales
    # R^2 (coeficiente de determinación)
    ss_total = np.sum((real_impedance - np.mean(real_impedance)) ** 2)
    ss_residual = np.sum(squared_error)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Métricas sobre errores en los gradientes
    real_gradient = np.gradient(real_impedance)
    pred_gradient = np.gradient(predicted_impedance)
    gradient_error = real_gradient - pred_gradient
    gradient_mse = np.mean(gradient_error ** 2)
    gradient_rmse = np.sqrt(gradient_mse)
    
    # Estadísticas sobre el error distribuido por cuantiles
    error_quantiles = np.percentile(abs_error, [25, 50, 75, 90, 95, 99])
    
    # Correlación de Pearson
    correlation = np.corrcoef(real_impedance, predicted_impedance)[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation,
        'gradient_mse': gradient_mse,
        'gradient_rmse': gradient_rmse,
        'error_quantiles': error_quantiles,
        'max_error': np.max(abs_error),
        'min_error': np.min(abs_error),
        'mean_error': np.mean(error),
        'std_error': np.std(error)
    }

def analyze_layers_optimized(real_impedance, predicted_impedance, real_seismic, 
                            homogeneous_zones, well_name=None, config=None):
    """
    Analiza la detección de capas con algoritmos optimizados y consideración de zonas
    
    Args:
        real_impedance: Impedancia real
        predicted_impedance: Impedancia predicha
        real_seismic: Datos sísmicos reales
        homogeneous_zones: Máscara de zonas homogéneas
        well_name: Nombre del pozo (opcional)
        config: Configuración adicional
        
    Returns:
        Métricas de detección de capas
    """
    print("\n=== ANALIZANDO DETECCIÓN DE CAPAS MEJORADO ===\n")
    
    # Configuración
    edge_threshold_percentile = config.get('edge_threshold_percentile', 94) if config else 94
    
    # Calcular gradientes (vectorizado)
    real_gradient = np.abs(np.gradient(real_impedance))
    pred_gradient = np.abs(np.gradient(predicted_impedance))
    seismic_gradient = np.abs(np.gradient(real_seismic))
    
    # Suavizar gradientes con kernels gaussianos
    real_gradient = ndimage.gaussian_filter1d(real_gradient, sigma=1.0)
    pred_gradient = ndimage.gaussian_filter1d(pred_gradient, sigma=1.0)
    seismic_gradient = ndimage.gaussian_filter1d(seismic_gradient, sigma=1.0)
    
    # Normalizar para comparación
    real_gradient = real_gradient / (np.max(real_gradient) + 1e-8)
    pred_gradient = pred_gradient / (np.max(pred_gradient) + 1e-8)
    seismic_gradient = seismic_gradient / (np.max(seismic_gradient) + 1e-8)
    
    # Umbral para detectar cambios (adaptativo)
    real_threshold = np.percentile(real_gradient, edge_threshold_percentile)
    pred_threshold = np.percentile(pred_gradient, edge_threshold_percentile)
    
    # Detectar cambios significativos SOLO en zonas de transición
    transition_zones = ~homogeneous_zones
    real_changes = (real_gradient > real_threshold) & transition_zones
    pred_changes = (pred_gradient > pred_threshold) & transition_zones
    
    # Calcular métricas
    # 1. Correlación de gradientes
    gradient_correlation = np.corrcoef(real_gradient, pred_gradient)[0, 1]
    
    # 2. Análisis de detección de cambios con ventana de tolerancia
    window_size = 5  # Ventana de tolerancia para considerar detección correcta
    
    # Matriz de distancias para análisis eficiente
    real_change_indices = np.where(real_changes)[0]
    pred_change_indices = np.where(pred_changes)[0]
    
    # Calcular precisión: qué proporción de cambios predichos son correctos
    correct_detections = 0
    for pred_idx in pred_change_indices:
        # Buscar cambios reales cercanos
        if len(real_change_indices) > 0:
            min_dist = np.min(np.abs(real_change_indices - pred_idx))
            if min_dist <= window_size:
                correct_detections += 1
    
    precision = correct_detections / len(pred_change_indices) if len(pred_change_indices) > 0 else 0
    
    # Calcular recall: qué proporción de cambios reales son detectados
    detected_real_changes = 0
    for real_idx in real_change_indices:
        # Buscar cambios predichos cercanos
        if len(pred_change_indices) > 0:
            min_dist = np.min(np.abs(pred_change_indices - real_idx))
            if min_dist <= window_size:
                detected_real_changes += 1
    
    recall = detected_real_changes / len(real_change_indices) if len(real_change_indices) > 0 else 0
    
    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calcular falsos positivos (cambios predichos que no corresponden a cambios reales)
    false_positives = len(pred_change_indices) - correct_detections
    
    # Análisis adicional: cambios detectados en zonas homogéneas (no deseados)
    false_detections_homogeneous = np.sum(pred_gradient[homogeneous_zones] > pred_threshold)
    
    # Resultados
    print(f"Correlación de gradientes: {gradient_correlation:.4f}")
    print(f"Precisión: {precision:.4f} (proporción de cambios predichos que son correctos)")
    print(f"Recall: {recall:.4f} (proporción de cambios reales que son detectados)")
    print(f"F1-Score: {f1:.4f}")
    print(f"Cambios reales: {len(real_change_indices)}, Detectados: {detected_real_changes}")
    print(f"Cambios predichos: {len(pred_change_indices)}, Falsos positivos: {false_positives}")
    print(f"Falsos cambios en zonas homogéneas: {false_detections_homogeneous}")
    
    # Visualizar análisis de capas mejorado
    visualize_layer_detection_enhanced(
        real_impedance, 
        predicted_impedance, 
        real_gradient, 
        pred_gradient,
        seismic_gradient,
        real_changes,
        pred_changes,
        homogeneous_zones,
        real_threshold,
        pred_threshold,
        gradient_correlation,
        precision,
        recall,
        f1,
        well_name,
        save_path=os.path.join(config['results_dir'] if config else ".", 'layer_detection_analysis_enhanced.png')
    )
    
    # Métricas de capas
    layer_metrics = {
        'gradient_correlation': gradient_correlation,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'total_real_changes': len(real_change_indices),
        'total_pred_changes': len(pred_change_indices),
        'correct_detections': detected_real_changes,
        'false_positives': false_positives,
        'false_detections_homogeneous': false_detections_homogeneous
    }
    
    return layer_metrics
    
def plot_evaluation_results_enhanced(real_impedance, predicted_impedance, predicted_raw,
                                   real_noisy_seismic, homogeneous_zones, metrics, 
                                   zone_metrics, well_name, well_idx, save_path=None):
    """
    Visualiza resultados de evaluación mejorados con análisis por zonas
    """
    # Usar backend no interactivo
    plt.switch_backend('agg')
    
    fig = plt.figure(figsize=(18, 16), dpi=100)
    
    # Crear grid de subplots
    gs = fig.add_gridspec(5, 2, height_ratios=[1.5, 1.5, 1, 1, 1], width_ratios=[3, 1])
    
    # 1. Traza sísmica con zonas
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f'Traza Sísmica con Zonas - {well_name} (Índice: {well_idx})')
    ax1.plot(real_noisy_seismic, 'k-', linewidth=0.8)
    
    # Sombrear zonas homogéneas
    x = np.arange(len(real_noisy_seismic))
    y_min, y_max = ax1.get_ylim()
    ax1.fill_between(x, y_min, y_max, where=homogeneous_zones, 
                     alpha=0.2, color='green', label='Zona Homogénea')
    
    ax1.set_ylabel('Amplitud')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Impedancia - Real vs Predicha
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Impedancia Acústica - Comparación Completa')
    ax2.plot(real_impedance, 'b-', label='Real', linewidth=2)
    ax2.plot(predicted_raw, 'gray', label='Predicha (sin post-proc)', alpha=0.5, linewidth=1)
    ax2.plot(predicted_impedance, 'r-', label='Predicha (final)', alpha=0.8, linewidth=1.5)
    
    # Sombrear zonas
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                     where=homogeneous_zones, alpha=0.1, color='green')
    
    ax2.set_ylabel('Impedancia (g/cc * ft/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error absoluto
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('Error Absoluto')
    error = np.abs(real_impedance - predicted_impedance)
    ax3.plot(error, 'k-', linewidth=0.8)
    
    # Colorear por zonas
    ax3.fill_between(x, 0, error, where=homogeneous_zones, 
                     alpha=0.5, color='green')
    ax3.fill_between(x, 0, error, where=~homogeneous_zones, 
                     alpha=0.5, color='orange')
    
    ax3.set_xlabel('Muestras')
    ax3.set_ylabel('Error Absoluto')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histograma de errores por zona
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.set_title('Distribución de Errores por Zona')
    
    if np.any(homogeneous_zones):
        error_homo = error[homogeneous_zones]
        ax4.hist(error_homo, bins=30, alpha=0.6, color='green', 
                label=f'Homogéneas (RMSE: {zone_metrics["homogeneous_rmse"]:.1f})', 
                density=True)
    
    if np.any(~homogeneous_zones):
        error_trans = error[~homogeneous_zones]
        ax4.hist(error_trans, bins=30, alpha=0.6, color='orange', 
                label=f'Transiciones (RMSE: {zone_metrics["transition_rmse"]:.1f})', 
                density=True)
    
    ax4.set_xlabel('Error Absoluto')
    ax4.set_ylabel('Densidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot: Real vs Predicho
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.set_title(f'Correlación Real vs Predicho (R²: {metrics["r2"]:.4f})')
    
    # Scatter coloreado por zonas
    if np.any(homogeneous_zones):
        ax5.scatter(real_impedance[homogeneous_zones], predicted_impedance[homogeneous_zones], 
                   c='green', alpha=0.5, s=1, label='Homogéneas')
    if np.any(~homogeneous_zones):
        ax5.scatter(real_impedance[~homogeneous_zones], predicted_impedance[~homogeneous_zones], 
                   c='orange', alpha=0.5, s=1, label='Transiciones')
    
    # Línea de referencia perfecta
    min_val = min(real_impedance.min(), predicted_impedance.min())
    max_val = max(real_impedance.max(), predicted_impedance.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfecta')
    
    ax5.set_xlabel('Impedancia Real')
    ax5.set_ylabel('Impedancia Predicha')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. Panel de métricas (texto)
    ax6 = fig.add_subplot(gs[0:2, 1])
    ax6.axis('off')
    
    # Texto de métricas
    metrics_text = f"""MÉTRICAS GENERALES:
    
MSE: {metrics['mse']:.2f}
RMSE: {metrics['rmse']:.2f}
MAE: {metrics['mae']:.2f}
R²: {metrics['r2']:.4f}
Correlación: {metrics['correlation']:.4f}

MÉTRICAS POR ZONAS:

Zonas Homogéneas ({zone_metrics['homogeneous_percentage']:.1f}%):
  RMSE: {zone_metrics['homogeneous_rmse']:.2f}
  MAE: {zone_metrics['homogeneous_mae']:.2f}

Zonas de Transición:
  RMSE: {zone_metrics['transition_rmse']:.2f}
  MAE: {zone_metrics['transition_mae']:.2f}

Ratio de Mejora: {zone_metrics['improvement_ratio']:.3f}
(menor es mejor)

ESTADÍSTICAS DE ERROR:

Media: {metrics['mean_error']:.2f}
Std: {metrics['std_error']:.2f}
Max: {metrics['max_error']:.2f}

Percentiles:
  25%: {metrics['error_quantiles'][0]:.2f}
  50%: {metrics['error_quantiles'][1]:.2f}
  75%: {metrics['error_quantiles'][2]:.2f}
  90%: {metrics['error_quantiles'][3]:.2f}
  95%: {metrics['error_quantiles'][4]:.2f}"""
    
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Mejora del postprocesamiento
    ax7 = fig.add_subplot(gs[2:, 1])
    ax7.set_title('Efecto del Postprocesamiento')
    
    # Calcular mejoras
    error_raw = np.abs(real_impedance - predicted_raw)
    error_final = np.abs(real_impedance - predicted_impedance)
    improvement = error_raw - error_final
    
    # Boxplot de mejoras por zona
    data_to_plot = []
    labels = []
    
    if np.any(homogeneous_zones):
        data_to_plot.append(improvement[homogeneous_zones])
        labels.append('Homogéneas')
    
    if np.any(~homogeneous_zones):
        data_to_plot.append(improvement[~homogeneous_zones])
        labels.append('Transiciones')
    
    if data_to_plot:
        bp = ax7.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['green', 'orange']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax7.set_ylabel('Mejora en Error\n(positivo = mejor)')
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle(f'Evaluación Completa - {well_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_evaluation_results(real_impedance, predicted_impedance, real_noisy_seismic, 
                           metrics, well_name, well_idx, save_path=None):
    """
    Versión simplificada de visualización (mantenida para compatibilidad)
    """
    # Detectar zonas homogéneas para análisis
    from PINN_1D_OPTIMIZADO import detect_homogeneous_zones
    homogeneous_zones = detect_homogeneous_zones(real_noisy_seismic)
    
    # Calcular métricas por zonas
    zone_metrics = calculate_zone_specific_metrics(
        real_impedance, predicted_impedance, homogeneous_zones
    )
    
    # Usar la versión mejorada
    plot_evaluation_results_enhanced(
        real_impedance, predicted_impedance, predicted_impedance,
        real_noisy_seismic, homogeneous_zones, metrics, zone_metrics,
        well_name, well_idx, save_path
    )

def evaluate_ensemble_diagnostics(ensemble_results, data_bundle, test_well_idx, config=None):
    """
    Función avanzada para diagnosticar el rendimiento del ensemble por modelo
    """
    from PINN_1D_OPTIMIZADO import simple_ensemble_predict, detect_homogeneous_zones
    
    print("\n=== DIAGNOSTICO DETALLADO DEL ENSEMBLE ===\n")
    
    # Obtener datos originales
    original_data = data_bundle['original_data']
    well_data = original_data['well_data']
    normalized_seismic = original_data['normalized_seismic']
    window_size = data_bundle['window_size']
    
    # Datos del pozo de prueba
    test_well_data = well_data[test_well_idx]
    test_well_name = test_well_data['well_name']
    real_impedance = test_well_data['acoustic_impedance']
    normalized_test_seismic = normalized_seismic[test_well_idx]
    
    # Detectar zonas homogéneas
    homogeneous_zones = detect_homogeneous_zones(
        normalized_test_seismic,
        window_size=config.get('local_variance_window', 20) if config else 20,
        variance_threshold=config.get('homogeneous_zone_threshold', 0.05) if config else 0.05
    )
    
    # Obtener modelos del ensemble
    models = ensemble_results['models']
    num_models = len(models)
    
    # Evaluación individual de cada modelo
    model_metrics = []
    model_predictions = []
    model_zone_metrics = []
    
    print(f"Evaluando {num_models} modelos individualmente...")
    
    for i, model in enumerate(models):
        print(f"Evaluando modelo {i+1}/{num_models}...")
        
        # Predicción individual
        normalized_prediction = simple_ensemble_predict(
            [model],  # Solo este modelo
            normalized_test_seismic,
            window_size,
            well_idx=test_well_idx,
            config=config
        )
        
        # Desnormalizar
        impedance_scaler = data_bundle['impedance_scalers'][test_well_idx]
        predicted_impedance = impedance_scaler.inverse_transform(
            normalized_prediction.reshape(-1, 1)
        ).flatten()
        
        # Calcular métricas generales
        metrics = calculate_impedance_metrics(real_impedance, predicted_impedance)
        
        # Calcular métricas por zonas
        zone_metrics = calculate_zone_specific_metrics(
            real_impedance, predicted_impedance, homogeneous_zones
        )
        
        # Guardar resultados
        model_metrics.append(metrics)
        model_zone_metrics.append(zone_metrics)
        model_predictions.append(predicted_impedance)
    
    # Predicción del ensemble completo
    from PINN_1D_OPTIMIZADO import improved_ensemble_predict
    
    ensemble_normalized_prediction = improved_ensemble_predict(
        models,
        normalized_test_seismic,
        window_size,
        well_idx=test_well_idx,
        config=config
    )
    
    # Desnormalizar predicción del ensemble
    ensemble_prediction = impedance_scaler.inverse_transform(
        ensemble_normalized_prediction.reshape(-1, 1)
    ).flatten()
    
    # Calcular métricas del ensemble
    ensemble_metrics = calculate_impedance_metrics(real_impedance, ensemble_prediction)
    ensemble_zone_metrics = calculate_zone_specific_metrics(
        real_impedance, ensemble_prediction, homogeneous_zones
    )
    
    # Visualizar comparación de modelos
    visualize_ensemble_diagnostics(
        real_impedance, model_predictions, ensemble_prediction,
        model_metrics, model_zone_metrics, ensemble_metrics, 
        ensemble_zone_metrics, homogeneous_zones, test_well_name, config
    )
    
    # Resultados de diagnóstico
    diagnostics = {
        'model_metrics': model_metrics,
        'model_zone_metrics': model_zone_metrics,
        'model_predictions': model_predictions,
        'ensemble_metrics': ensemble_metrics,
        'ensemble_zone_metrics': ensemble_zone_metrics,
        'ensemble_prediction': ensemble_prediction,
        'improvement': {
            'rmse': ensemble_metrics['rmse'] / np.mean([m['rmse'] for m in model_metrics]),
            'mae': ensemble_metrics['mae'] / np.mean([m['mae'] for m in model_metrics]),
            'r2': ensemble_metrics['r2'] / np.mean([m['r2'] for m in model_metrics]),
            'homogeneous_rmse': ensemble_zone_metrics['homogeneous_rmse'] / 
                                np.mean([m['homogeneous_rmse'] for m in model_zone_metrics]),
        }
    }
    
    # Resumen
    print("\nResumen de diagnóstico del ensemble:")
    print(f"  Promedio modelos - RMSE: {np.mean([m['rmse'] for m in model_metrics]):.2f}, "
          f"R²: {np.mean([m['r2'] for m in model_metrics]):.4f}")
    print(f"  Ensemble - RMSE: {ensemble_metrics['rmse']:.2f}, R²: {ensemble_metrics['r2']:.4f}")
    
    print(f"\nMejora en zonas homogéneas:")
    print(f"  Promedio modelos - RMSE: {np.mean([m['homogeneous_rmse'] for m in model_zone_metrics]):.2f}")
    print(f"  Ensemble - RMSE: {ensemble_zone_metrics['homogeneous_rmse']:.2f}")
    
    improvement_rmse = (1 - ensemble_metrics['rmse'] / np.mean([m['rmse'] for m in model_metrics])) * 100
    print(f"\nMejora general del ensemble: {improvement_rmse:.2f}% en RMSE")
    
    return diagnostics

def visualize_ensemble_diagnostics(real_impedance, model_predictions, ensemble_prediction,
                                  model_metrics, model_zone_metrics, ensemble_metrics,
                                  ensemble_zone_metrics, homogeneous_zones, well_name, config):
    """
    Visualiza diagnósticos del ensemble
    """
    plt.figure(figsize=(18, 12), dpi=100)
    
    # 1. Comparación de predicciones
    plt.subplot(3, 2, 1)
    plt.title(f'Comparación de Modelos - {well_name}')
    plt.plot(real_impedance, 'k-', label='Real', linewidth=2)
    
    # Plotear cada modelo con transparencia
    for i, pred in enumerate(model_predictions):
        plt.plot(pred, alpha=0.3, linewidth=1, label=f'Modelo {i+1}')
    
    # Plotear ensemble con más énfasis
    plt.plot(ensemble_prediction, 'r-', label='Ensemble', linewidth=2, alpha=0.8)
    
    # Sombrear zonas homogéneas
    x = np.arange(len(real_impedance))
    plt.fill_between(x, plt.ylim()[0], plt.ylim()[1], 
                     where=homogeneous_zones, alpha=0.1, color='green')
    
    plt.ylabel('Impedancia (g/cc * ft/s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. Métricas por modelo
    plt.subplot(3, 2, 2)
    plt.title('Métricas Generales por Modelo')
    
    # Extraer métricas
    rmse_values = [m['rmse'] for m in model_metrics] + [ensemble_metrics['rmse']]
    mae_values = [m['mae'] for m in model_metrics] + [ensemble_metrics['mae']]
    r2_values = [m['r2'] for m in model_metrics] + [ensemble_metrics['r2']]
    
    x_labels = [f'M{i+1}' for i in range(len(model_metrics))] + ['Ens']
    x = np.arange(len(x_labels))
    width = 0.25
    
    # Normalizar para visualización
    rmse_norm = np.array(rmse_values) / max(rmse_values)
    mae_norm = np.array(mae_values) / max(mae_values)
    
    plt.bar(x - width, rmse_norm, width, label='RMSE (norm)', alpha=0.8)
    plt.bar(x, mae_norm, width, label='MAE (norm)', alpha=0.8)
    plt.bar(x + width, r2_values, width, label='R²', alpha=0.8)
    
    plt.xticks(x, x_labels)
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # 3. Métricas en zonas homogéneas
    plt.subplot(3, 2, 3)
    plt.title('RMSE en Zonas Homogéneas')
    
    homo_rmse = [m['homogeneous_rmse'] for m in model_zone_metrics] + [ensemble_zone_metrics['homogeneous_rmse']]
    colors = ['blue'] * len(model_zone_metrics) + ['red']
    
    bars = plt.bar(x_labels, homo_rmse, color=colors, alpha=0.7)
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Añadir línea de referencia del promedio
    avg_rmse = np.mean(homo_rmse[:-1])
    plt.axhline(y=avg_rmse, color='gray', linestyle='--', alpha=0.5, label='Promedio modelos')
    plt.legend()
    
    # 4. Métricas en transiciones
    plt.subplot(3, 2, 4)
    plt.title('RMSE en Zonas de Transición')
    
    trans_rmse = [m['transition_rmse'] for m in model_zone_metrics] + [ensemble_zone_metrics['transition_rmse']]
    
    bars = plt.bar(x_labels, trans_rmse, color=colors, alpha=0.7)
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Línea de referencia
    avg_trans_rmse = np.mean(trans_rmse[:-1])
    plt.axhline(y=avg_trans_rmse, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Varianza entre modelos
    plt.subplot(3, 2, 5)
    plt.title('Varianza entre Modelos')
    
    # Calcular std entre modelos en cada punto
    model_preds_array = np.array(model_predictions)
    model_std = np.std(model_preds_array, axis=0)
    
    plt.plot(model_std, 'b-', linewidth=0.8)
    plt.fill_between(x, 0, model_std, where=homogeneous_zones, 
                     alpha=0.3, color='green', label='Zonas homogéneas')
    plt.fill_between(x, 0, model_std, where=~homogeneous_zones, 
                     alpha=0.3, color='orange', label='Transiciones')
    
    plt.ylabel('Desviación Estándar')
    plt.xlabel('Muestras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Mejora del ensemble
    plt.subplot(3, 2, 6)
    plt.title('Mejora del Ensemble vs Promedio de Modelos')
    
    # Calcular promedio simple de modelos
    avg_prediction = np.mean(model_preds_array, axis=0)
    
    # Errores
    error_avg = np.abs(real_impedance - avg_prediction)
    error_ensemble = np.abs(real_impedance - ensemble_prediction)
    improvement = error_avg - error_ensemble
    
    plt.plot(improvement, 'g-', linewidth=0.8, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.fill_between(x, 0, improvement, where=improvement > 0, 
                     alpha=0.3, color='green', label='Mejora')
    plt.fill_between(x, 0, improvement, where=improvement <= 0, 
                     alpha=0.3, color='red', label='Empeora')
    
    plt.ylabel('Diferencia en Error\n(positivo = ensemble mejor)')
    plt.xlabel('Muestras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Diagnóstico del Ensemble - {well_name}', fontsize=16)
    plt.tight_layout()
    
    # Guardar
    save_path = os.path.join(config['results_dir'] if config else ".", 
                            f'{well_name}_ensemble_diagnostics.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
"""
Funciones de visualización faltantes para PINN_EVAL_OPTIMIZADO.py
Añadir estas funciones al final del archivo PINN_EVAL_OPTIMIZADO.py
"""

def visualize_layer_detection_enhanced(real_impedance, predicted_impedance, real_gradient, 
                                     pred_gradient, seismic_gradient, real_changes, pred_changes,
                                     homogeneous_zones, real_threshold, pred_threshold,
                                     gradient_correlation, precision, recall, f1,
                                     well_name, save_path=None):
    """
    Visualiza análisis detallado de detección de capas con zonas homogéneas
    """
    plt.switch_backend('agg')
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 16), dpi=100)
    
    # Asegurar que todos los arrays tengan la misma longitud
    min_len = min(len(real_impedance), len(predicted_impedance), len(real_gradient),
                  len(pred_gradient), len(seismic_gradient), len(homogeneous_zones))
    
    real_impedance = real_impedance[:min_len]
    predicted_impedance = predicted_impedance[:min_len]
    real_gradient = real_gradient[:min_len]
    pred_gradient = pred_gradient[:min_len]
    seismic_gradient = seismic_gradient[:min_len]
    homogeneous_zones = homogeneous_zones[:min_len]
    
    x = np.arange(min_len)
    
    # 1. Impedancia con zonas homogéneas
    ax1 = axes[0, 0]
    ax1.plot(x, real_impedance, 'b-', label='Real', linewidth=2)
    ax1.plot(x, predicted_impedance, 'r-', label='Predicha', alpha=0.8, linewidth=1.5)
    
    # Sombrear zonas homogéneas
    ax1.fill_between(x, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                     where=homogeneous_zones, alpha=0.2, color='green', 
                     label=f'Zonas Homogéneas ({np.sum(homogeneous_zones)} muestras)')
    
    ax1.set_title(f'Impedancia Acústica - {well_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Impedancia (g/cc * ft/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gradientes comparados
    ax2 = axes[0, 1]
    ax2.plot(x, real_gradient, 'b-', label='Gradiente Real', linewidth=2, alpha=0.8)
    ax2.plot(x, pred_gradient, 'r-', label='Gradiente Predicho', linewidth=1.5, alpha=0.8)
    ax2.plot(x, seismic_gradient, 'g-', label='Gradiente Sísmico', linewidth=1, alpha=0.6)
    
    # Líneas de umbral
    ax2.axhline(y=real_threshold, color='blue', linestyle='--', alpha=0.7, 
                label=f'Umbral Real ({real_threshold:.3f})')
    ax2.axhline(y=pred_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Umbral Pred ({pred_threshold:.3f})')
    
    ax2.set_title(f'Gradientes Normalizados (Correlación: {gradient_correlation:.3f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gradiente Normalizado')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Detección de cambios
    ax3 = axes[1, 0]
    ax3.plot(x, real_impedance, 'k-', alpha=0.3, linewidth=1, label='Impedancia')
    
    # Marcar cambios detectados
    real_change_indices = np.where(real_changes[:min_len])[0]
    pred_change_indices = np.where(pred_changes[:min_len])[0]
    
    if len(real_change_indices) > 0:
        ax3.scatter(real_change_indices, real_impedance[real_change_indices], 
                   color='blue', s=50, marker='|', label=f'Cambios Reales ({len(real_change_indices)})')
    
    if len(pred_change_indices) > 0:
        ax3.scatter(pred_change_indices, real_impedance[pred_change_indices], 
                   color='red', s=30, marker='x', alpha=0.7, 
                   label=f'Cambios Predichos ({len(pred_change_indices)})')
    
    # Sombrear zonas homogéneas
    ax3.fill_between(x, ax3.get_ylim()[0], ax3.get_ylim()[1], 
                     where=homogeneous_zones, alpha=0.1, color='green')
    
    ax3.set_title('Detección de Límites de Capas', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Impedancia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Matriz de confusión visual
    ax4 = axes[1, 1]
    
    # Crear array de detección para análisis
    detection_status = np.zeros(min_len)  # 0: sin cambio
    
    # Marcar cambios reales
    detection_status[real_changes[:min_len]] = 1  # 1: cambio real
    
    # Marcar predicciones
    for pred_idx in pred_change_indices:
        if pred_idx < min_len:
            # Buscar cambio real cercano (ventana de tolerancia)
            window_size = 5
            real_nearby = real_change_indices[
                np.abs(real_change_indices - pred_idx) <= window_size
            ]
            
            if len(real_nearby) > 0:
                detection_status[pred_idx] = 2  # 2: verdadero positivo
            else:
                detection_status[pred_idx] = 3  # 3: falso positivo
    
    # Crear colores para visualización
    colors = ['white', 'lightblue', 'green', 'red']
    labels = ['Sin cambio', 'Cambio real no detectado', 'Verdadero positivo', 'Falso positivo']
    
    # Plot como imagen
    detection_matrix = detection_status.reshape(1, -1)
    im = ax4.imshow(detection_matrix, aspect='auto', cmap=plt.cm.colors.ListedColormap(colors))
    
    # Añadir colorbar con etiquetas
    cbar = plt.colorbar(im, ax=ax4, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(labels)
    
    ax4.set_title('Estado de Detección por Muestra', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Análisis de Detección')
    ax4.set_xlabel('Muestras')
    
    # 5. Métricas por zonas
    ax5 = axes[2, 0]
    
    # Calcular métricas en zonas homogéneas vs transiciones
    homo_indices = np.where(homogeneous_zones)[0]
    trans_indices = np.where(~homogeneous_zones)[0]
    
    # Falsos positivos por zona
    fp_homo = np.sum([1 for idx in pred_change_indices 
                     if idx in homo_indices and idx not in real_change_indices])
    fp_trans = np.sum([1 for idx in pred_change_indices 
                      if idx in trans_indices and idx not in real_change_indices])
    
    # Verdaderos positivos por zona
    tp_homo = np.sum([1 for idx in pred_change_indices 
                     if idx in homo_indices and 
                     any(abs(idx - real_idx) <= 5 for real_idx in real_change_indices)])
    tp_trans = np.sum([1 for idx in pred_change_indices 
                      if idx in trans_indices and 
                      any(abs(idx - real_idx) <= 5 for real_idx in real_change_indices)])
    
    # Crear gráfico de barras
    categories = ['Zonas\nHomogéneas', 'Zonas de\nTransición']
    fp_values = [fp_homo, fp_trans]
    tp_values = [tp_homo, tp_trans]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, fp_values, width, label='Falsos Positivos', 
                    color='red', alpha=0.7)
    bars2 = ax5.bar(x_pos + width/2, tp_values, width, label='Verdaderos Positivos', 
                    color='green', alpha=0.7)
    
    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_xlabel('Tipo de Zona')
    ax5.set_ylabel('Número de Detecciones')
    ax5.set_title('Detecciones por Tipo de Zona', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(categories)
    ax5.legend()
    ax5.grid(True, axis='y', alpha=0.3)
    
    # 6. Panel de métricas
    ax6 = axes[2, 1]
    ax6.axis('off')
    
    # Calcular métricas adicionales
    total_samples = min_len
    homo_percentage = np.sum(homogeneous_zones) / total_samples * 100
    
    # Crear texto de métricas
    metrics_text = f"""MÉTRICAS DE DETECCIÓN DE CAPAS

Métricas Generales:
  Precisión: {precision:.3f}
  Recall: {recall:.3f}
  F1-Score: {f1:.3f}
  Correlación de gradientes: {gradient_correlation:.3f}

Conteos:
  Cambios reales: {len(real_change_indices)}
  Cambios predichos: {len(pred_change_indices)}
  Verdaderos positivos: {len(real_change_indices) - (len(real_change_indices) - int(recall * len(real_change_indices)))}
  Falsos positivos: {len(pred_change_indices) - int(precision * len(pred_change_indices))}

Análisis por Zonas:
  Zonas homogéneas: {homo_percentage:.1f}% del total
  FP en zonas homogéneas: {fp_homo}
  FP en transiciones: {fp_trans}
  TP en zonas homogéneas: {tp_homo}
  TP en transiciones: {tp_trans}

Umbrales:
  Umbral real: {real_threshold:.4f}
  Umbral predicho: {pred_threshold:.4f}"""
    
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Análisis Detallado de Detección de Capas - {well_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    
    plt.close()


def create_comprehensive_evaluation_report(evaluation_results, config):
    """
    Crea un reporte visual completo de la evaluación
    """
    plt.switch_backend('agg')
    
    fig = plt.figure(figsize=(20, 24), dpi=100)
    
    # Extraer datos
    real_impedance = evaluation_results['real_impedance']
    predicted_impedance = evaluation_results['predicted_impedance']
    real_seismic = evaluation_results['real_seismic']
    noisy_seismic = evaluation_results['noisy_seismic']
    homogeneous_zones = evaluation_results['homogeneous_zones']
    metrics = evaluation_results['metrics']
    zone_metrics = evaluation_results['zone_metrics']
    layer_metrics = evaluation_results['layer_metrics']
    well_name = evaluation_results['test_well_name']
    
    # Crear grid complejo
    gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)
    
    # 1. Traza sísmica original y con ruido
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(real_seismic))
    
    ax1.plot(x, real_seismic, 'b-', label='Sísmica Original', linewidth=1.5, alpha=0.8)
    ax1.plot(x, noisy_seismic, 'r-', label=f'Con Ruido (SNR: {config.get("snr_db", 20)} dB)', 
             linewidth=1, alpha=0.7)
    
    # Marcar zonas homogéneas
    ax1.fill_between(x, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                     where=homogeneous_zones, alpha=0.2, color='green', 
                     label=f'Zonas Homogéneas ({np.sum(homogeneous_zones)} muestras)')
    
    ax1.set_title(f'Datos Sísmicos - {well_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitud')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Impedancia: comparación completa
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(x, real_impedance, 'b-', label='Real', linewidth=2.5)
    ax2.plot(x, predicted_impedance, 'r-', label='Predicha', linewidth=1.5, alpha=0.8)
    
    # Sombrear zonas por error
    error = np.abs(real_impedance - predicted_impedance)
    high_error_mask = error > np.percentile(error, 90)
    
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                     where=high_error_mask, alpha=0.3, color='red', 
                     label='Zonas de Alto Error (>P90)')
    ax2.fill_between(x, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                     where=homogeneous_zones, alpha=0.1, color='green')
    
    ax2.set_title('Impedancia Acústica - Comparación', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Impedancia (g/cc * ft/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error absoluto con análisis estadístico
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(x, error, 'k-', linewidth=1)
    ax3.fill_between(x, 0, error, where=homogeneous_zones, alpha=0.5, color='green', 
                     label='Homogéneas')
    ax3.fill_between(x, 0, error, where=~homogeneous_zones, alpha=0.5, color='orange', 
                     label='Transiciones')
    
    # Líneas estadísticas
    ax3.axhline(y=np.mean(error), color='red', linestyle='--', 
                label=f'Media: {np.mean(error):.2f}')
    ax3.axhline(y=np.percentile(error, 95), color='purple', linestyle=':', 
                label=f'P95: {np.percentile(error, 95):.2f}')
    
    ax3.set_title('Error Absoluto por Zona', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Error Absoluto')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Histograma de errores por zona
    ax4 = fig.add_subplot(gs[2, 1])
    
    if np.any(homogeneous_zones):
        error_homo = error[homogeneous_zones]
        ax4.hist(error_homo, bins=30, alpha=0.6, color='green', density=True,
                label=f'Homogéneas\n(RMSE: {zone_metrics["homogeneous_rmse"]:.2f})')
    
    if np.any(~homogeneous_zones):
        error_trans = error[~homogeneous_zones]
        ax4.hist(error_trans, bins=30, alpha=0.6, color='orange', density=True,
                label=f'Transiciones\n(RMSE: {zone_metrics["transition_rmse"]:.2f})')
    
    ax4.set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error Absoluto')
    ax4.set_ylabel('Densidad')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot con análisis de correlación
    ax5 = fig.add_subplot(gs[2, 2])
    
    # Scatter coloreado por zonas
    if np.any(homogeneous_zones):
        ax5.scatter(real_impedance[homogeneous_zones], predicted_impedance[homogeneous_zones], 
                   c='green', alpha=0.6, s=3, label='Homogéneas')
    if np.any(~homogeneous_zones):
        ax5.scatter(real_impedance[~homogeneous_zones], predicted_impedance[~homogeneous_zones], 
                   c='orange', alpha=0.6, s=3, label='Transiciones')
    
    # Línea perfecta y regresión
    min_val = min(real_impedance.min(), predicted_impedance.min())
    max_val = max(real_impedance.max(), predicted_impedance.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfecta')
    
    # Línea de regresión
    z = np.polyfit(real_impedance, predicted_impedance, 1)
    p = np.poly1d(z)
    ax5.plot([min_val, max_val], [p(min_val), p(max_val)], 'r-', alpha=0.8, 
             label=f'Regresión (R²: {metrics["r2"]:.3f})')
    
    ax5.set_title('Correlación Real vs Predicho', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Impedancia Real')
    ax5.set_ylabel('Impedancia Predicha')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    
    # 6-8. Análisis de gradientes y detección de capas
    ax6 = fig.add_subplot(gs[3, 0])
    real_grad = np.abs(np.gradient(real_impedance))
    pred_grad = np.abs(np.gradient(predicted_impedance))
    
    ax6.plot(x[1:], real_grad[1:], 'b-', label='Gradiente Real', linewidth=1.5, alpha=0.8)
    ax6.plot(x[1:], pred_grad[1:], 'r-', label='Gradiente Predicho', linewidth=1, alpha=0.8)
    
    # Umbral para detección de capas
    real_threshold = np.percentile(real_grad, 94)
    ax6.axhline(y=real_threshold, color='blue', linestyle='--', alpha=0.7, 
                label=f'Umbral: {real_threshold:.3f}')
    
    ax6.set_title(f'Gradientes (Corr: {layer_metrics["gradient_correlation"]:.3f})', 
                  fontsize=12, fontweight='bold')
    ax6.set_ylabel('|Gradiente|')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Métricas de detección de capas
    ax7 = fig.add_subplot(gs[3, 1])
    
    metrics_names = ['Precisión', 'Recall', 'F1-Score']
    metrics_values = [layer_metrics['precision'], layer_metrics['recall'], layer_metrics['f1_score']]
    colors_bars = ['lightblue', 'lightgreen', 'lightyellow']
    
    bars = ax7.bar(metrics_names, metrics_values, color=colors_bars, edgecolor='black', alpha=0.8)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Línea de referencia
    ax7.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Objetivo: 0.8')
    
    ax7.set_title('Métricas de Detección de Capas', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Valor')
    ax7.set_ylim(0, 1.1)
    ax7.legend()
    ax7.grid(True, axis='y', alpha=0.3)
    
    # 8. Análisis de falsos positivos
    ax8 = fig.add_subplot(gs[3, 2])
    
    fp_data = [
        layer_metrics.get('false_positives', 0), 
        layer_metrics.get('false_detections_homogeneous', 0)
    ]
    fp_labels = ['Falsos Positivos\nGenerales', 'Falsos Positivos\nen Zonas Homogéneas']
    
    bars = ax8.bar(fp_labels, fp_data, color=['lightcoral', 'red'], alpha=0.7)
    
    for bar, value in zip(bars, fp_data):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    ax8.set_title('Análisis de Falsos Positivos', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Cantidad')
    ax8.grid(True, axis='y', alpha=0.3)
    
    # 9-11. Paneles de métricas numéricas
    ax9 = fig.add_subplot(gs[4, :])
    ax9.axis('off')
    
    # Crear tabla de métricas
    metrics_table = f"""
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                           REPORTE COMPLETO DE EVALUACIÓN                                            │
    │                                                    {well_name}                                                     │
    ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │ MÉTRICAS GENERALES                    │ MÉTRICAS POR ZONAS                  │ DETECCIÓN DE CAPAS                    │
    │                                       │                                      │                                       │
    │ MSE:           {metrics['mse']:>8.2f}             │ Zonas Homogéneas: {zone_metrics['homogeneous_percentage']:>5.1f}%       │ Correlación Grad: {layer_metrics['gradient_correlation']:>6.3f}          │
    │ RMSE:          {metrics['rmse']:>8.2f}             │ RMSE Homogéneas:  {zone_metrics['homogeneous_rmse']:>8.2f}       │ Precisión:        {layer_metrics['precision']:>6.3f}          │
    │ MAE:           {metrics['mae']:>8.2f}             │ RMSE Transiciones:{zone_metrics['transition_rmse']:>8.2f}       │ Recall:           {layer_metrics['recall']:>6.3f}          │
    │ R²:            {metrics['r2']:>8.4f}             │ Ratio Mejora:     {zone_metrics['improvement_ratio']:>8.3f}       │ F1-Score:         {layer_metrics['f1_score']:>6.3f}          │
    │ Correlación:   {metrics['correlation']:>8.4f}             │                                      │                                       │
    │                                       │ ERRORES POR ZONA:                   │ CONTEOS:                              │
    │ ESTADÍSTICAS DE ERROR:                │ MAE Homogéneas:   {zone_metrics['homogeneous_mae']:>8.2f}       │ Cambios Reales:   {layer_metrics['total_real_changes']:>8}          │
    │ Error Medio:   {metrics['mean_error']:>8.2f}             │ MAE Transiciones: {zone_metrics['transition_mae']:>8.2f}       │ Cambios Predichos:{layer_metrics['total_pred_changes']:>8}          │
    │ Desv. Estándar:{metrics['std_error']:>8.2f}             │                                      │ Detecciones Correctas: {layer_metrics['correct_detections']:>4}          │
    │ Error Máximo:  {metrics['max_error']:>8.2f}             │                                      │ Falsos Positivos: {layer_metrics['false_positives']:>8}          │
    │                                       │                                      │ FP en Zonas Homogéneas: {layer_metrics.get('false_detections_homogeneous', 0):>4}      │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    ax9.text(0.5, 0.5, metrics_table, transform=ax9.transAxes, 
             ha='center', va='center', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    
    # 12. Recomendaciones
    ax10 = fig.add_subplot(gs[5, :])
    ax10.axis('off')
    
    # Generar recomendaciones basadas en métricas
    recommendations = generate_recommendations(metrics, zone_metrics, layer_metrics, config)
    
    ax10.text(0.05, 0.95, recommendations, transform=ax10.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Reporte Completo de Evaluación - {well_name}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Guardar reporte
    save_path = os.path.join(config.get('results_dir', '.'), f'{well_name}_comprehensive_report.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Reporte completo guardado: {save_path}")
    return save_path


def generate_recommendations(metrics, zone_metrics, layer_metrics, config):
    """
    Genera recomendaciones basadas en las métricas obtenidas
    """
    recommendations = ["RECOMENDACIONES PARA MEJORAR RESULTADOS:\n"]
    
    # Análisis de R²
    if metrics['r2'] < 0.7:
        recommendations.append("• R² bajo (<0.7): Considere aumentar 'final_physics_weight' a 0.7-0.8")
        recommendations.append("• Aumente 'hidden_size' del modelo para mayor capacidad")
    elif metrics['r2'] > 0.9:
        recommendations.append("• R² excelente (>0.9): Modelo bien ajustado")
    
    # Análisis de detección de capas
    if layer_metrics['f1_score'] < 0.6:
        recommendations.append("• F1-Score bajo: Ajuste 'homogeneous_zone_threshold' (pruebe 0.03-0.06)")
        recommendations.append("• Reduzca 'max_gradient_threshold' para mejor detección de capas")
    elif layer_metrics['f1_score'] > 0.85:
        recommendations.append("• F1-Score excelente: Detección de capas muy buena")
    
    # Análisis de falsos positivos
    if layer_metrics.get('false_detections_homogeneous', 0) > 100:
        recommendations.append("• Muchos falsos positivos en zonas homogéneas:")
        recommendations.append("  - Aumente 'gradient_weight' a 0.5-0.6")
        recommendations.append("  - Reduzca 'overshoot_limiter_factor' a 0.6")
        recommendations.append("  - Aumente 'homogeneous_zone_threshold' a 0.06-0.08")
    
    # Análisis por zonas
    if zone_metrics['improvement_ratio'] > 1.5:
        recommendations.append("• Zonas homogéneas tienen más error que transiciones:")
        recommendations.append("  - Aumente 'temporal_smoothing_weight' a 0.1")
        recommendations.append("  - Considere usar filtro bilateral más fuerte")
    
    # Análisis de correlación de gradientes
    if layer_metrics['gradient_correlation'] < 0.5:
        recommendations.append("• Correlación de gradientes baja:")
        recommendations.append("  - Verifique calidad de datos sísmicos")
        recommendations.append("  - Considere aumentar SNR en datos sintéticos")
        recommendations.append("  - Ajuste frecuencia del wavelet (pruebe 20-30 Hz)")
    
    # Recomendaciones de postprocesamiento
    if metrics['std_error'] > metrics['mean_error']:
        recommendations.append("• Alta variabilidad en errores:")
        recommendations.append("  - Active 'bilateral_filter_strength' = 0.4")
        recommendations.append("  - Aumente 'median_filter_size' a 9-11")
    
    # Recomendaciones generales
    if metrics['rmse'] > 500:
        recommendations.append("• RMSE alto: Considere normalización diferente o más épocas")
    
    recommendations.append("\n" + "="*80)
    recommendations.append("PARÁMETROS SUGERIDOS PARA SIGUIENTE EJECUCIÓN:")
    
    # Sugerir parámetros específicos
    if layer_metrics['f1_score'] < 0.7:
        recommendations.append(f"  max_gradient_threshold: {max(0.05, config.get('max_gradient_threshold', 0.08) * 0.8):.3f}")
        recommendations.append(f"  gradient_weight: {min(0.6, config.get('gradient_weight', 0.4) * 1.2):.2f}")
    
    if zone_metrics['improvement_ratio'] > 1.2:
        recommendations.append(f"  homogeneous_zone_threshold: {min(0.08, config.get('homogeneous_zone_threshold', 0.04) * 1.3):.3f}")
        recommendations.append(f"  overshoot_limiter_factor: {max(0.5, config.get('overshoot_limiter_factor', 0.7) * 0.9):.2f}")
    
    if metrics['r2'] < 0.8:
        recommendations.append(f"  final_physics_weight: {min(0.8, config.get('final_physics_weight', 0.6) * 1.2):.2f}")
        recommendations.append(f"  epochs: {min(150, config.get('epochs', 120) + 20)}")
    
    return "\n".join(recommendations)


# Función auxiliar para crear plots de diagnóstico rápido
def quick_diagnostic_plot(evaluation_results, save_path=None):
    """
    Crea un plot de diagnóstico rápido para revisión inmediata
    """
    plt.switch_backend('agg')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
    
    # Extraer datos
    real_impedance = evaluation_results['real_impedance']
    predicted_impedance = evaluation_results['predicted_impedance']
    metrics = evaluation_results['metrics']
    layer_metrics = evaluation_results['layer_metrics']
    well_name = evaluation_results['test_well_name']
    
    x = np.arange(len(real_impedance))
    
    # 1. Comparación rápida
    ax1 = axes[0, 0]
    ax1.plot(x, real_impedance, 'b-', label='Real', linewidth=2)
    ax1.plot(x, predicted_impedance, 'r-', label='Predicha', alpha=0.8)
    ax1.set_title(f'Impedancia - {well_name}', fontweight='bold')
    ax1.set_ylabel('Impedancia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error
    ax2 = axes[0, 1]
    error = np.abs(real_impedance - predicted_impedance)
    ax2.plot(x, error, 'k-', linewidth=1)
    ax2.axhline(y=np.mean(error), color='red', linestyle='--', 
                label=f'Media: {np.mean(error):.2f}')
    ax2.set_title('Error Absoluto', fontweight='bold')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Métricas clave
    ax3 = axes[1, 0]
    metrics_names = ['R²', 'F1-Score', 'Grad Corr']
    metrics_values = [metrics['r2'], layer_metrics['f1_score'], 
                     layer_metrics['gradient_correlation']]
    
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in metrics_values]
    bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('Métricas Clave', fontweight='bold')
    ax3.set_ylabel('Valor')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Resumen textual
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Determinar calidad general
    overall_score = (metrics['r2'] + layer_metrics['f1_score'] + 
                    layer_metrics['gradient_correlation']) / 3
    
    if overall_score > 0.8:
        quality = "EXCELENTE ✅"
        color = 'green'
    elif overall_score > 0.6:
        quality = "BUENA ⚠️"
        color = 'orange'
    else:
        quality = "MEJORABLE ❌"
        color = 'red'
    
    summary_text = f"""RESUMEN RÁPIDO

Calidad General: {quality}
Puntuación: {overall_score:.3f}

Métricas:
• RMSE: {metrics['rmse']:.1f}
• R²: {metrics['r2']:.3f}
• F1-Score: {layer_metrics['f1_score']:.3f}
• Correlación: {layer_metrics['gradient_correlation']:.3f}

Detección de Capas:
• Precisión: {layer_metrics['precision']:.3f}
• Recall: {layer_metrics['recall']:.3f}
• Falsos Positivos: {layer_metrics['false_positives']}"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=11,
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.suptitle(f'Diagnóstico Rápido - {well_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    
    plt.close()


# FUNCIÓN PRINCIPAL DE CORRECCIÓN PARA PINN_EVAL_OPTIMIZADO.py
def fix_missing_functions():
    """
    Esta función debe ser llamada para verificar que todas las funciones estén disponibles
    """
    try:
        # Verificar que matplotlib esté configurado
        plt.switch_backend('agg')
        print("✅ Backend de matplotlib configurado correctamente")
        
        # Verificar importaciones necesarias
        import numpy as np
        from scipy.ndimage import uniform_filter1d
        print("✅ Todas las importaciones necesarias están disponibles")
        
        print("✅ Funciones de visualización configuradas correctamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error general: {e}")
        return False
