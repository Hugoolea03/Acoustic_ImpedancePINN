"""
PINN_MAIN_OPTIMIZADO.py - Script Principal con Todas las Optimizaciones Integradas

Características:
- Flujo completo optimizado para PyTorch y GPU
- Configuración centralizada para todos los módulos
- Estructura modular para fácil mantenimiento
"""
import os
import time
import datetime
import argparse
import numpy as np
import torch
import warnings
import random
warnings.filterwarnings('ignore')

# Verificar disponibilidad de GPU y configurar para rendimiento óptimo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Permitir TF32 en operaciones de matrices
    torch.backends.cudnn.allow_tf32 = True  # Permitir TF32 en convolutions

# Configuración global para todo el proyecto - OPTIMIZADA PARA MEJOR PRECISIÓN
CONFIG = {
    # Parámetros existentes (mantener)
    'results_dir': None,
    'data_path': None,
    'las_pattern': "SYNTHETIC_WELL_*.las",
    'num_wells': 'all',
    
    # Parámetros de datos (sin cambios)
    'freq': 30,           
    'length': 0.512,      
    'dt': 0.002,          
    'snr_db': 25,         
    'window_size': 48,    
    'use_weighting': True,
    
    # Parámetros del modelo (sin cambios)
    'hidden_size': 512,   
    'dropout_rate': 0.15, 
    
    # Parámetros de entrenamiento (sin cambios)
    'batch_size': 24,     
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'epochs': 60,         
    'k_folds': 3,        
    
    # Parámetros de predicción (sin cambios)
    'num_models_ensemble': 3,
    'stride_factor': 10,
    
    # ===== PARÁMETROS CRÍTICOS PARA REDUCIR SPIKES ===== #
    
    # 1. PÉRDIDA FÍSICA MÁS AGRESIVA (CAMBIO MÁS IMPORTANTE)
    'initial_physics_weight': 0.25,    # AUMENTADO de 0.18 -> 0.25
    'final_physics_weight': 0.70,     # AUMENTADO de 0.45 -> 0.70 (MUY IMPORTANTE)
    'epochs_to_increase': 20,          # REDUCIDO de 25 -> 20 (más rápido)
    
    # 2. CONTROL DE GRADIENTES MÁS RESTRICTIVO (CRÍTICO)
    'max_gradient_threshold': 0.08,    # REDUCIDO de 0.12 -> 0.08 (más restrictivo)
    'gradient_weight': 0.40,           # AUMENTADO de 0.25 -> 0.40 (más penalización)
    
    # 3. DETECCIÓN DE ZONAS HOMOGÉNEAS MÁS SENSIBLE (MUY IMPORTANTE)
    'homogeneous_zone_threshold': 0.03,  # REDUCIDO de 0.05 -> 0.03 (detecta más zonas)
    'local_variance_window': 25,         # AUMENTADO de 20 -> 25 (mejor contexto)
    
    # 4. LIMITACIÓN DE SOBREPASOS MÁS AGRESIVA (CRÍTICO)
    'overshoot_limiter_factor': 0.6,    # REDUCIDO de 0.8 -> 0.6 (más restrictivo)
    
    # 5. DETECCIÓN DE BORDES MENOS SENSIBLE (PARA PRESERVAR ZONAS HOMOGÉNEAS)
    'edge_threshold_percentile': 96,    # AUMENTADO de 94 -> 96 (menos bordes falsos)
    
    # Parámetros de optimización (sin cambios)
    'lr_scheduler_factor': 0.3,
    'lr_scheduler_patience': 8,
    'early_stopping_patience': 15,
    
    # 6. FILTRADO DE OUTLIERS MÁS ESTRICTO (IMPORTANTE)
    'ensemble_outlier_threshold': 1.5,  # REDUCIDO de 2.0 -> 1.5 (más estricto)
    
    # Parámetros específicos para pozo
    'use_well_specific_processing': True,
}

# ===== CONFIGURACIÓN AVANZADA ADICIONAL ===== #
ADVANCED_CONFIG = {
    # 1. PARÁMETROS PARA PÉRDIDA ADAPTATIVA MÁS AGRESIVA
    'homogeneous_penalty_factor': 5.0,      # AUMENTADO de 3.0 -> 5.0 (MUY IMPORTANTE)
    'transition_penalty_factor': 0.3,       # REDUCIDO de 0.4 -> 0.3 (menos en transiciones)
    
    # 2. SUAVIZADO TEMPORAL MÁS FUERTE (CRÍTICO PARA ZONAS HOMOGÉNEAS)
    'temporal_smoothing_weight': 0.15,      # AUMENTADO de 0.08 -> 0.15 (más suavizado)
    
    # 3. FILTRO BILATERAL MÁS FUERTE (NUEVO - MUY IMPORTANTE)
    'bilateral_filter_strength': 0.50,      # AUMENTADO de 0.35 -> 0.50
    
    # 4. PARÁMETROS DE DETECCIÓN DE SPIKES MÁS SENSIBLES
    'spike_detection_threshold': 2.0,       # REDUCIDO de 2.5 -> 2.0 (más sensible)
    'homogeneous_smoothing_factor': 2.0,    # AUMENTADO de 1.5 -> 2.0 (más suavizado)
    
    # 5. PRESERVACIÓN DE TRANSICIONES REALES (IMPORTANTE)
    'transition_preservation_factor': 0.8,  # AUMENTADO de 0.7 -> 0.8
    
    # 6. PARÁMETROS PARA VALORES ALTOS (SIN CAMBIOS CRÍTICOS)
    'high_value_boost': 1.1,               # REDUCIDO de 1.2 -> 1.1 (menos agresivo)
    'dynamic_scaling_threshold': 50000,     
    'adaptive_window_sizing': True,         
}

# ===== PARÁMETROS ESPECÍFICOS PARA EnhancedSpikeControlLoss ===== #
# Estos son los más importantes para tu problema específico
SPIKE_CONTROL_CONFIG = {
    # Control de gradientes más estricto
    'max_gradient_threshold': 0.06,        # MUY RESTRICTIVO (era 0.12)
    'gradient_weight': 0.50,               # MUY ALTO (era 0.25)
    
    # Penalización muy alta en zonas homogéneas
    'homogeneous_penalty_factor': 6.0,     # MUY ALTO (era 3.0)
    'transition_penalty_factor': 0.2,      # MUY BAJO (era 0.4)
    
    # Detección de spikes más sensible
    'spike_detection_threshold': 1.8,      # MUY SENSIBLE (era 2.5)
    
    # Suavizado temporal muy fuerte
    'temporal_smoothing_weight': 0.20,     # MUY ALTO (era 0.08)
    
    # Boost menos agresivo para valores altos
    'high_value_boost': 1.05,              # CASI NEUTRO (era 1.2)
    'dynamic_scaling_threshold': 45000,    # LIGERAMENTE REDUCIDO
}

# ===== FUNCIÓN PARA APLICAR LA CONFIGURACIÓN OPTIMIZADA ===== #
def get_optimized_config_for_spike_reduction():
    """
    Devuelve configuración optimizada específicamente para reducir spikes
    en zonas homogéneas y mejorar detección de transiciones
    """
    # Combinar todas las configuraciones
    optimized_config = CONFIG.copy()
    optimized_config.update(ADVANCED_CONFIG)
    optimized_config.update(SPIKE_CONTROL_CONFIG)
    
    return optimized_config

# NUEVOS PARÁMETROS ADICIONALES PARA CONTROL FINO
ADVANCED_CONFIG = {
    # Parámetros para pérdida adaptativa mejorada
    'homogeneous_penalty_factor': 3.0,      # NUEVO: Mayor penalización en zonas homogéneas
    'transition_penalty_factor': 0.4,       # NUEVO: Menor penalización en transiciones válidas
    'temporal_smoothing_weight': 0.08,      # NUEVO: Suavizado temporal más fuerte
    'bilateral_filter_strength': 0.35,     # NUEVO: Filtro bilateral para preservar bordes
    
    # Parámetros para detección de zonas mejorada
    'spike_detection_threshold': 2.5,      # NUEVO: Umbral para detectar spikes
    'homogeneous_smoothing_factor': 1.5,   # NUEVO: Suavizado extra en zonas homogéneas
    'transition_preservation_factor': 0.7, # NUEVO: Preservar transiciones reales
    
    # Parámetros para mejores predicciones en valores altos
    'high_value_boost': 1.2,              # NUEVO: Amplificar predicciones en valores altos
    'dynamic_scaling_threshold': 50000,    # NUEVO: Umbral para escalado dinámico
    'adaptive_window_sizing': True,        # NUEVO: Ventanas adaptativas por zona
}

# FUNCIÓN PARA APLICAR CONFIGURACIÓN AVANZADA
def apply_advanced_config(base_config):
    """
    Aplica configuración avanzada al config base
    """
    enhanced_config = base_config.copy()
    enhanced_config.update(ADVANCED_CONFIG)
    return enhanced_config

def main():
    """Función principal de ejecución con flujo optimizado"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Inversión Acústica con PINNs - Versión Optimizada')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='/home/hugo/Escritorio/Geophysical_Inversion/DATA_WELL_LOGS/GP_HPO_Paper-main/HPO_Code/Data/LAS')
    parser.add_argument('--las_pattern', type=str, default="SYNTHETIC_WELL_*.las",
                        help='Patrón para buscar archivos LAS')
    parser.add_argument('--num_wells', type=str, default='all',
                        help='Número de pozos a utilizar (o "all" para todos)')
    parser.add_argument('--window_size', type=int, default=64,
                        help='Tamaño de ventana para el contexto')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Número de épocas para entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño de lote para entrenamiento')
    parser.add_argument('--test_well', type=int, default=None,
                        help='Índice del pozo a usar para prueba (None para aleatorio)')
    parser.add_argument('--snr_db', type=float, default=20,
                        help='Relación señal-ruido en dB para datos sintéticos')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Tamaño de capas ocultas en el modelo')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Tasa de aprendizaje inicial')
    parser.add_argument('--physics_weight', type=float, default=0.45,
                        help='Peso final para la pérdida física')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Deshabilitar el uso de CUDA')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directorio para guardar resultados')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Solo visualizar resultados existentes sin entrenar')
    parser.add_argument('--ensemble_diagnostics', action='store_true',
                        help='Generar diagnóstico detallado del ensemble')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para reproducibilidad')
    
    args = parser.parse_args()
    
    # Establecer semilla para reproducibilidad
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Actualizar configuración con argumentos
    CONFIG['data_path'] = args.data_path
    CONFIG['las_pattern'] = args.las_pattern
    CONFIG['window_size'] = args.window_size
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    CONFIG['snr_db'] = args.snr_db
    CONFIG['hidden_size'] = args.hidden_size
    CONFIG['learning_rate'] = args.lr
    CONFIG['final_physics_weight'] = args.physics_weight
    CONFIG['initial_physics_weight'] = args.physics_weight * 0.4  # Inicio más bajo
    
    # Convertir num_wells a entero si no es 'all'
    if args.num_wells != 'all':
        CONFIG['num_wells'] = int(args.num_wells)
    else:
        CONFIG['num_wells'] = 'all'
    
    # Configurar el dispositivo
    if args.no_cuda:
        global device
        device = torch.device("cpu")
        print("Forzando uso de CPU")
    
    # Crear directorio de resultados
    if args.results_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        CONFIG['results_dir'] = f"resultados_inversion_optimizado_{timestamp}"
    else:
        CONFIG['results_dir'] = args.results_dir
    
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Guardar configuración para referencia
    with open(os.path.join(CONFIG['results_dir'], 'config.txt'), 'w') as f:
        for key, value in CONFIG.items():
            f.write(f"{key}: {value}\n")
    
    # Iniciar cronómetro
    start_time = time.time()
    
    # Información del sistema
    print("\n" + "="*60)
    print(f"INVERSIÓN ACÚSTICA CON PINNS - VERSIÓN OPTIMIZADA")
    print(f"Dispositivo: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available() and not args.no_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Ejecutando en CPU")
    print("="*60 + "\n")
    
    try:
        # Importar funciones optimizadas
        from PINN_TRAIN_OPTIMIZADO import prepare_data_optimized, create_ensemble_optimized
        from PINN_EVAL_OPTIMIZADO import evaluate_model_optimized, analyze_layers_optimized, evaluate_ensemble_diagnostics
        
        # Preparar datos con optimizaciones
        data_bundle = prepare_data_optimized(CONFIG)
        
        if not args.visualize_only:
            # Crear ensemble de modelos
            ensemble_results = create_ensemble_optimized(data_bundle, CONFIG)
            
            # Evaluar modelo
            evaluation_results = evaluate_model_optimized(
                ensemble_results,
                data_bundle,
                test_well_idx=args.test_well,
                config=CONFIG
            )
            
            # Análisis adicional: diagnóstico detallado del ensemble
            if args.ensemble_diagnostics:
                test_well_idx = evaluation_results['test_well_idx']
                diagnostics = evaluate_ensemble_diagnostics(
                    ensemble_results,
                    data_bundle,
                    test_well_idx,
                    config=CONFIG
                )
        else:
            # Modo visualización: cargar resultados previos
            import glob
            results_files = glob.glob(os.path.join(CONFIG['results_dir'], '*_results.npy'))
            
            if not results_files:
                raise FileNotFoundError("No se encontraron resultados previos para visualizar.")
            
            latest_results = max(results_files, key=os.path.getctime)
            print(f"Cargando resultados previos: {latest_results}")
            
            evaluation_results = np.load(latest_results, allow_pickle=True).item()
            
            # Analizar detección de capas con datos existentes
            layer_metrics = analyze_layers_optimized(
                evaluation_results['real_impedance'],
                evaluation_results['predicted_impedance'],
                evaluation_results['real_seismic'],
                well_name=evaluation_results['test_well_name'],
                config=CONFIG
            )
            
            # Actualizar resultados con nuevas métricas de capas
            evaluation_results['layer_metrics'] = layer_metrics
        
        # Tiempo total
        total_duration = time.time() - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        # Resultados finales
        print("\n" + "="*60)
        print(f'RESULTADOS FINALES:')
        print(f'MSE: {evaluation_results["metrics"]["mse"]:.2f}')
        print(f'RMSE: {evaluation_results["metrics"]["rmse"]:.2f}')
        print(f'MAE: {evaluation_results["metrics"]["mae"]:.2f}')
        print(f'R²: {evaluation_results["metrics"]["r2"]:.4f}')
        print(f'F1-Score (detección de capas): {evaluation_results["layer_metrics"]["f1_score"]:.4f}')
        print(f'Correlación de gradientes: {evaluation_results["layer_metrics"]["gradient_correlation"]:.4f}')
        print(f'Pozo de prueba: {evaluation_results["test_well_name"]}')
        print(f'Resultados guardados en: {CONFIG["results_dir"]}')
        print(f'Tiempo total: {hours}h {minutes}m {seconds}s')
        print("="*60 + "\n")
        
        # Guardar informe final
        with open(os.path.join(CONFIG['results_dir'], "resultados_finales.txt"), 'w') as f:
            f.write("="*60 + "\n")
            f.write("INVERSIÓN ACÚSTICA CON PINNS - RESULTADOS FINALES\n")
            f.write("="*60 + "\n\n")
            f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dispositivo: {device}\n")
            f.write(f"Pozo de prueba: {evaluation_results['test_well_name']}\n\n")
            f.write("MÉTRICAS DE ERROR:\n")
            f.write(f"MSE: {evaluation_results['metrics']['mse']:.2f}\n")
            f.write(f"RMSE: {evaluation_results['metrics']['rmse']:.2f}\n")
            f.write(f"MAE: {evaluation_results['metrics']['mae']:.2f}\n")
            f.write(f"R²: {evaluation_results['metrics']['r2']:.4f}\n\n")
            f.write("MÉTRICAS DE DETECCIÓN DE CAPAS:\n")
            f.write(f"F1-Score: {evaluation_results['layer_metrics']['f1_score']:.4f}\n")
            f.write(f"Precisión: {evaluation_results['layer_metrics']['precision']:.4f}\n")
            f.write(f"Recall: {evaluation_results['layer_metrics']['recall']:.4f}\n")
            f.write(f"Correlación de gradientes: {evaluation_results['layer_metrics']['gradient_correlation']:.4f}\n")
            f.write(f"Total cambios reales: {evaluation_results['layer_metrics']['total_real_changes']}\n")
            f.write(f"Total cambios predichos: {evaluation_results['layer_metrics']['total_pred_changes']}\n")
            f.write(f"Detecciones correctas: {evaluation_results['layer_metrics']['correct_detections']}\n")
            f.write(f"Falsos positivos: {evaluation_results['layer_metrics']['false_positives']}\n\n")
            f.write("CONFIGURACIÓN:\n")
            for key, value in CONFIG.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nTiempo total: {hours}h {minutes}m {seconds}s\n")
            f.write("="*60 + "\n")
        
        print(f"Informe guardado en: {os.path.join(CONFIG['results_dir'], 'resultados_finales.txt')}")
        
        # Mostrar sugerencias adicionales
        print("\nSugerencias para mejorar resultados:")
        if evaluation_results['layer_metrics']['f1_score'] < 0.6:
            print("- La detección de capas puede mejorar ajustando 'homogeneous_zone_threshold'")
            print("- Considere aumentar 'local_variance_window' para mejor análisis contextual")
        
        if evaluation_results['metrics']['r2'] < 0.75:
            print("- Para mejorar R², considere aumentar 'final_physics_weight' a 0.5")
            print("- Ajuste 'overshoot_limiter_factor' si hay picos excesivos")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        
        # Guardar error
        with open(os.path.join(CONFIG['results_dir'], "error_log.txt"), 'w') as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())
            
        print(f"El error se ha registrado en: {os.path.join(CONFIG['results_dir'], 'error_log.txt')}")

if __name__ == "__main__":
    main()
