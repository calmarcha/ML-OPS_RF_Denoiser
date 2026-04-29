"""
RF Denoiser — Script principal.

Ejecuta el pipeline completo:
  1. Configuración (W&B login, detección de dispositivo)
  2. Carga y preprocesado de datos de entrenamiento
  3. Creación del Dataset (train / val split)
  4. Entrenamiento de GRU, LSTM, CRN y Transformer
  5. Análisis de resultados de entrenamiento (tiem2pos, curvas de loss)
  6. Medición de tiempos de inferencia
  7. Evaluación en dataset de test (voz no vista)
  8. Prueba de audio con el mejor modelo (espectrogramas)
  9. Guardar resultados en CSV y cerrar runs de W&B

Uso:
    cd src
    python main.py
"""

import sys
import warnings
from pathlib import Path

# Asegura que la carpeta src/ esté en el path cuando se lanza desde la raíz
_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd
import torch
import wandb
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Imports del proyecto ──────────────────────────────────────────────────────
from config import (
    NOISE_FILE, VOICE_FILE,
    TEST_NOISE_FILE, TEST_VOICE_FILE,
    RESULTS_DIR, WANDB_PROJECT,
    device,
)
from logging_config import setup_wandb
from data import build_datasets, build_test_dataset
from model_GRU         import GRUDenoiser
from model_LSTM        import LSTMDenoiser
from model_CRN         import CRNDenoiser
from model_transformer import TransformerDenoiser
from train      import train_model
from evaluate   import measure_inference_time, evaluate_on_test
from visualize  import (
    plot_training_times,
    plot_loss_curves,
    plot_inference_times,
    plot_val_vs_test,
    plot_spectrograms,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 1 — Configuración
# ─────────────────────────────────────────────────────────────────────────────
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA no disponible, usando CPU")

setup_wandb()


# ─────────────────────────────────────────────────────────────────────────────
# Paso 2-3 — Datos de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
train_dataset, val_dataset = build_datasets(VOICE_FILE, NOISE_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 4-6 — Entrenar todos los modelos
# ─────────────────────────────────────────────────────────────────────────────
results: dict = {}

for model_cls, model_name in [
    (GRUDenoiser,         'GRU'),
    (LSTMDenoiser,        'LSTM'),
    (CRNDenoiser,         'CRN'),
    (TransformerDenoiser, 'Transformer'),
]:
    model = model_cls()
    results[model_name] = train_model(model, model_name, train_dataset, val_dataset)

print("\n" + "="*60)
print("TODOS LOS MODELOS ENTRENADOS")
print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 7 — Análisis de resultados de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

# 7.1 Tiempos de entrenamiento
print("\nTIEMPOS DE ENTRENAMIENTO")
times_df = pd.DataFrame(
    [(name, r['training_time']) for name, r in results.items()],
    columns=['Modelo', 'Tiempo (s)'],
).sort_values('Tiempo (s)')
print(times_df.to_string(index=False))
plot_training_times(results)

# 7.2 Curvas de loss
plot_loss_curves(results)

# 7.3 Tabla resumen
summary_data = []
for name, result in results.items():
    val_metrics = result['metrics'][['val_loss']].dropna()
    if len(val_metrics) > 0:
        summary_data.append({
            'Modelo':         name,
            'Final Val Loss': f"{val_metrics['val_loss'].iloc[-1]:.6f}",
            'Best Val Loss':  f"{val_metrics['val_loss'].min():.6f}",
            'Tiempo (s)':     f"{result['training_time']:.2f}",
        })
print("\n" + "="*100)
print("RESUMEN DE RESULTADOS (ENTRENAMIENTO)")
print("="*100)
print(pd.DataFrame(summary_data).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Paso 8 — Tiempos de inferencia
# ─────────────────────────────────────────────────────────────────────────────
inference_times: dict = {}
for name, result in results.items():
    t = measure_inference_time(result['model'])
    inference_times[name] = t
    print(f"{name}: {t:.3f} ms")

plot_inference_times(inference_times)

# Registrar en W&B
real_time_ms = 2000.0
for name, t in inference_times.items():
    ratio = real_time_ms / t
    results[name]['wandb_logger'].experiment.log({
        'inference_time_ms': t,
        'realtime_ratio':    ratio,
    })
print("✓ Tiempos de inferencia registrados en W&B")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 8.5 — Evaluación en dataset de test
# ─────────────────────────────────────────────────────────────────────────────
test_dataset, _ = build_test_dataset(TEST_VOICE_FILE, TEST_NOISE_FILE)

print("\n" + "="*60)
print("EVALUACIÓN EN DATASET DE TEST")
print("="*60)

test_results: dict = {}
for name, result in results.items():
    metrics = evaluate_on_test(result['model'], test_dataset)
    test_results[name] = metrics
    print(
        f"{name:15s}: Test Loss = {metrics['test_loss']:.6f} | "
        f"MSE = {metrics['test_mse']:.6f} | RMSE = {metrics['test_rmse']:.6f}"
    )

plot_val_vs_test(results, test_results)

# Registrar en W&B
for name in results:
    val_loss  = results[name]['metrics']['val_loss'].dropna().min()
    test_loss = test_results[name]['test_loss']
    gap       = ((test_loss - val_loss) / val_loss) * 100
    results[name]['wandb_logger'].experiment.log({
        'test_l1_loss':    test_results[name]['test_loss'],
        'test_mse':        test_results[name]['test_mse'],
        'test_rmse':       test_results[name]['test_rmse'],
        'val_test_gap_pct': gap,
    })
print("✓ Métricas de test registradas en W&B")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 9 — Prueba de audio con el mejor modelo
# ─────────────────────────────────────────────────────────────────────────────
best_model_name = min(
    results.keys(),
    key=lambda x: results[x]['metrics']['val_loss'].dropna().min(),
)
print(f"\n{'='*60}")
print(f"MEJOR MODELO: {best_model_name}")
print(f"{'='*60}\n")

best_model = results[best_model_name]['model']
best_model.eval()
best_model = best_model.to(device)

noisy_test, clean_test = test_dataset[0]
with torch.no_grad():
    denoised_output = best_model(noisy_test.unsqueeze(0).to(device))
    denoised_output = denoised_output.squeeze().cpu().numpy()

noisy_spec = noisy_test.squeeze().numpy()
clean_spec = clean_test.squeeze().numpy()

plot_spectrograms(noisy_spec, denoised_output, clean_spec, best_model_name)

mse  = np.mean((denoised_output - clean_spec) ** 2)
rmse = np.sqrt(mse)
print(f"\nMétricas de calidad en muestra de test:")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 10 — Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
results_path = Path(RESULTS_DIR)
results_path.mkdir(exist_ok=True)

# Tabla resumen
summary_with_test = []
for name, result in results.items():
    val_metrics = result['metrics'][['val_loss']].dropna()
    if len(val_metrics) > 0:
        summary_with_test.append({
            'Modelo':                  name,
            'Best Val Loss':           f"{val_metrics['val_loss'].min():.6f}",
            'Test Loss':               f"{test_results[name]['test_loss']:.6f}",
            'Test RMSE':               f"{test_results[name]['test_rmse']:.6f}",
            'Tiempo Entrenamiento (s)': f"{result['training_time']:.2f}",
            'Tiempo Inferencia (ms)':  f"{inference_times[name]:.3f}",
        })

summary_full_df = pd.DataFrame(summary_with_test)
summary_full_df.to_csv(results_path / 'model_comparison_summary.csv', index=False)
print(f"✓ Resumen guardado en {results_path / 'model_comparison_summary.csv'}")

# Tiempos de inferencia
pd.DataFrame(
    list(inference_times.items()), columns=['Modelo', 'Tiempo_Inferencia_ms']
).to_csv(results_path / 'inference_times.csv', index=False)
print(f"✓ Tiempos de inferencia guardados en {results_path / 'inference_times.csv'}")

# Resultados de test
pd.DataFrame([
    {'Modelo': n, 'Test_Loss': v['test_loss'], 'Test_MSE': v['test_mse'], 'Test_RMSE': v['test_rmse']}
    for n, v in test_results.items()
]).to_csv(results_path / 'test_results.csv', index=False)
print(f"✓ Resultados de test guardados en {results_path / 'test_results.csv'}")

# Métricas individuales
for name, result in results.items():
    out_file = results_path / f'{name}_metrics.csv'
    result['metrics'].to_csv(out_file, index=False)
    print(f"✓ Métricas de {name} guardadas en {out_file}")

# Tabla comparativa en W&B y cierre de runs
summary_table = wandb.Table(dataframe=summary_full_df)
for name, result in results.items():
    result['wandb_logger'].experiment.log({'comparison_table': summary_table})
    result['wandb_logger'].experiment.finish()
    print(f"✓ Run W&B finalizado para {name}")

print(f"\n{'='*60}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*60}")
print(f"\nResultados guardados en: {results_path.absolute()}")
print(f"Runs de W&B disponibles en: https://wandb.ai/ — Proyecto '{WANDB_PROJECT}'")
