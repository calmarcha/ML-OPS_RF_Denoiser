"""
Visualización de resultados: curvas de pérdida, tiempos, espectrogramas.

Funciones públicas:
    plot_training_times      — Gráfico de barras con tiempos de entrenamiento.
    plot_loss_curves         — Curvas train/val loss por modelo.
    plot_inference_times     — Gráfico de barras con tiempos de inferencia.
    plot_val_vs_test         — Comparativa Best Val Loss vs Test Loss.
    plot_spectrograms        — Tres espectrogramas en dB: ruidoso/denoised/limpio.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import matplotlib.pyplot as plt
import librosa

from config import HOP_LENGTH, SAMPLE_RATE, FIGURES_DIR

_FIGURES_DIR = Path(FIGURES_DIR)
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Colores por modelo ────────────────────────────────────────────────────────
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_training_times(results: dict) -> None:
    """Gráfico de barras con tiempos totales de entrenamiento."""
    times = {name: r['training_time'] for name, r in results.items()}
    times_sorted = dict(sorted(times.items(), key=lambda x: x[1]))

    plt.figure(figsize=(10, 6))
    plt.bar(times_sorted.keys(), times_sorted.values(), color=_COLORS)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Tiempo de Entrenamiento (segundos)', fontsize=12)
    plt.title('Comparación de Tiempos de Entrenamiento', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / 'training_times.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_loss_curves(results: dict) -> None:
    """Curvas de train_loss y val_loss por época para todos los modelos."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for name, result in results.items():
        metrics = result['metrics']
        train_loss = metrics[['epoch', 'train_loss']].dropna()
        plt.plot(train_loss['epoch'], train_loss['train_loss'], label=name, linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.title('Pérdida en Entrenamiento', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, result in results.items():
        metrics = result['metrics']
        val_loss = metrics[['epoch', 'val_loss']].dropna()
        plt.plot(val_loss['epoch'], val_loss['val_loss'], label=name, linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Pérdida en Validación', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_inference_times(inference_times: dict, real_time_ms: float = 2000.0) -> None:
    """Gráfico de barras con tiempos de inferencia y análisis de tiempo real."""
    models_list = list(inference_times.keys())
    times_list  = list(inference_times.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models_list, times_list, color=_COLORS)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Tiempo de Inferencia (ms)', fontsize=12)
    plt.title('Tiempo de Inferencia por Modelo', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, max(times_list) * 1.3)

    for bar, t in zip(bars, times_list):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times_list) * 0.02,
            f'{t:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / 'inference_times.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("ANÁLISIS DE TIEMPO REAL")
    print("="*60)
    for name, t in inference_times.items():
        ratio  = real_time_ms / t
        status = "✓ VIABLE" if ratio > 1 else "✗ NO VIABLE"
        print(f"{name:15s}: {t:7.3f} ms | Ratio: {ratio:6.2f}x | {status}")
    print()


def plot_val_vs_test(results: dict, test_results: dict) -> None:
    """Gráfico de barras comparando Best Val Loss vs Test Loss."""
    model_names     = list(results.keys())
    val_losses_list = [results[n]['metrics']['val_loss'].dropna().min() for n in model_names]
    test_losses_list = [test_results[n]['test_loss'] for n in model_names]

    x     = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, val_losses_list,  width, label='Best Val Loss',  color='#1f77b4', alpha=0.8)
    bars2 = plt.bar(x + width/2, test_losses_list, width, label='Test Loss',      color='#ff7f0e', alpha=0.8)

    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Loss (L1)', fontsize=12)
    plt.title('Comparación: Validación vs Test (Datos No Vistos)', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / 'val_vs_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*80)
    print("RESUMEN TEST - GENERALIZACIÓN A DATOS NO VISTOS")
    print("="*80)
    for name in model_names:
        val_loss  = results[name]['metrics']['val_loss'].dropna().min()
        test_loss = test_results[name]['test_loss']
        gap       = ((test_loss - val_loss) / val_loss) * 100
        print(f"{name:15s}: Val={val_loss:.6f} | Test={test_loss:.6f} | Gap={gap:+.1f}%")


def plot_spectrograms(
    noisy_spec:    np.ndarray,
    denoised_spec: np.ndarray,
    clean_spec:    np.ndarray,
    model_name:    str,
) -> None:
    """Visualiza tres espectrogramas en escala dB: ruidoso, denoised y limpio."""
    noisy_db    = librosa.amplitude_to_db(noisy_spec,    ref=np.max)
    denoised_db = librosa.amplitude_to_db(denoised_spec, ref=np.max)
    clean_db    = librosa.amplitude_to_db(clean_spec,    ref=np.max)

    duration_sec  = noisy_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE
    max_freq_khz  = SAMPLE_RATE / 2000  # Nyquist en kHz
    extent        = [0, duration_sec, 0, max_freq_khz]

    plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(3, 1, 1)
    img1 = ax1.imshow(noisy_db, aspect='auto', origin='lower', cmap='viridis',
                      vmin=-80, vmax=0, extent=extent)
    plt.colorbar(img1, format='%+2.0f dB')
    ax1.set_title('Espectrograma con Ruido (Test Data)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frecuencia (kHz)', fontsize=12)

    ax2 = plt.subplot(3, 1, 2)
    img2 = ax2.imshow(denoised_db, aspect='auto', origin='lower', cmap='viridis',
                      vmin=-80, vmax=0, extent=extent)
    plt.colorbar(img2, format='%+2.0f dB')
    ax2.set_title(f'Espectrograma Procesado ({model_name})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frecuencia (kHz)', fontsize=12)

    ax3 = plt.subplot(3, 1, 3)
    img3 = ax3.imshow(clean_db, aspect='auto', origin='lower', cmap='viridis',
                      vmin=-80, vmax=0, extent=extent)
    plt.colorbar(img3, format='%+2.0f dB')
    ax3.set_title('Espectrograma Limpio (Ground Truth)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Frecuencia (kHz)', fontsize=12)
    ax3.set_xlabel('Tiempo (s)', fontsize=12)

    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / f'spectrograms_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
