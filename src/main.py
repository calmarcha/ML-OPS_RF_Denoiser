"""
RF Denoiser — Script principal (Transformer).

Ejecuta el pipeline completo:
  1. Configuración (W&B login, detección de dispositivo)
  2. Carga y preprocesado de datos de entrenamiento
  3. Creación del Dataset (train / val split)
  4. Entrenamiento del modelo Transformer
  5. Análisis de resultados de entrenamiento (curvas de loss)
  6. Medición de tiempos de inferencia
  7. Evaluación en dataset de test (voz no vista)
  8. Prueba de audio (espectrogramas)
  9. Guardar resultados en CSV y cerrar run de W&B

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
from model_transformer import TransformerDenoiser
from train    import train_model
from evaluate import measure_inference_time, evaluate_on_test
from visualize import (
    plot_loss_curves,
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
# Paso 4 — Entrenar el modelo Transformer
# ─────────────────────────────────────────────────────────────────────────────
model = TransformerDenoiser()
result = train_model(model, 'Transformer', train_dataset, val_dataset)

print("\n" + "="*60)
print("TRANSFORMER ENTRENADO")
print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 5 — Análisis de resultados de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nTIEMPO DE ENTRENAMIENTO: {result['training_time']:.2f} segundos")

val_metrics = result['metrics'][['val_loss']].dropna()
if len(val_metrics) > 0:
    print(f"Best Val Loss:  {val_metrics['val_loss'].min():.6f}")
    print(f"Final Val Loss: {val_metrics['val_loss'].iloc[-1]:.6f}")

plot_loss_curves(result)


# ─────────────────────────────────────────────────────────────────────────────
# Paso 6 — Tiempos de inferencia
# ─────────────────────────────────────────────────────────────────────────────
inference_time_ms = measure_inference_time(result['model'])
real_time_ms = 2000.0
realtime_ratio = real_time_ms / inference_time_ms

print(f"\nTiempo de inferencia: {inference_time_ms:.3f} ms")
print(f"Ratio tiempo real (2 s): {realtime_ratio:.2f}x")

result['wandb_logger'].experiment.log({
    'inference_time_ms': inference_time_ms,
    'realtime_ratio':    realtime_ratio,
})
print("✓ Tiempo de inferencia registrado en W&B")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 7 — Evaluación en dataset de test
# ─────────────────────────────────────────────────────────────────────────────
test_dataset, _ = build_test_dataset(TEST_VOICE_FILE, TEST_NOISE_FILE)

print("\n" + "="*60)
print("EVALUACIÓN EN DATASET DE TEST")
print("="*60)

test_metrics = evaluate_on_test(result['model'], test_dataset)
print(
    f"Transformer: Test Loss = {test_metrics['test_loss']:.6f} | "
    f"MSE = {test_metrics['test_mse']:.6f} | RMSE = {test_metrics['test_rmse']:.6f}"
)

val_loss  = result['metrics']['val_loss'].dropna().min()
test_loss = test_metrics['test_loss']
gap       = ((test_loss - val_loss) / val_loss) * 100
print(f"Val/Test gap: {gap:+.1f}%")

result['wandb_logger'].experiment.log({
    'test_l1_loss':     test_metrics['test_loss'],
    'test_mse':         test_metrics['test_mse'],
    'test_rmse':        test_metrics['test_rmse'],
    'val_test_gap_pct': gap,
})
print("✓ Métricas de test registradas en W&B")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 8 — Prueba de audio (espectrogramas)
# ─────────────────────────────────────────────────────────────────────────────
transformer = result['model']
transformer.eval()
transformer = transformer.to(device)

noisy_test, clean_test = test_dataset[0]
with torch.no_grad():
    denoised_output = transformer(noisy_test.unsqueeze(0).to(device))
    denoised_output = denoised_output.squeeze().cpu().numpy()

noisy_spec = noisy_test.squeeze().numpy()
clean_spec = clean_test.squeeze().numpy()

plot_spectrograms(noisy_spec, denoised_output, clean_spec, 'Transformer')

mse  = float(np.mean((denoised_output - clean_spec) ** 2))
rmse = float(np.sqrt(mse))
print(f"\nMétricas de calidad en muestra de test:")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Paso 9 — Guardar resultados
# ─────────────────────────────────────────────────────────────────────────────
results_path = Path(RESULTS_DIR)
results_path.mkdir(exist_ok=True)

# Métricas de entrenamiento
out_metrics = results_path / 'Transformer_metrics.csv'
result['metrics'].to_csv(out_metrics, index=False)
print(f"✓ Métricas de entrenamiento guardadas en {out_metrics}")

# Resultados de test
pd.DataFrame([{
    'Modelo':           'Transformer',
    'Test_Loss':        test_metrics['test_loss'],
    'Test_MSE':         test_metrics['test_mse'],
    'Test_RMSE':        test_metrics['test_rmse'],
    'Val_Test_Gap_pct': gap,
}]).to_csv(results_path / 'test_results.csv', index=False)
print(f"✓ Resultados de test guardados en {results_path / 'test_results.csv'}")

# Tiempos de inferencia
pd.DataFrame([{
    'Modelo':               'Transformer',
    'Tiempo_Inferencia_ms':  inference_time_ms,
    'Ratio_Tiempo_Real':     realtime_ratio,
}]).to_csv(results_path / 'inference_times.csv', index=False)
print(f"✓ Tiempos de inferencia guardados en {results_path / 'inference_times.csv'}")

# Cerrar run de W&B
result['wandb_logger'].experiment.finish()
print("✓ Run W&B finalizado")

print(f"\n{'='*60}")
print("ANÁLISIS COMPLETADO")
print(f"{'='*60}")
print(f"\nResultados guardados en: {results_path.absolute()}")
print(f"Run W&B disponible en: https://wandb.ai/ — Proyecto '{WANDB_PROJECT}'")
