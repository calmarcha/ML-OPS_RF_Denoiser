"""
Visualización de resultados: curvas de pérdida y espectrogramas.

Funciones públicas:
    plot_loss_curves  — Curvas train/val loss del modelo Transformer.
    plot_spectrograms — Tres espectrogramas en dB: ruidoso/denoised/limpio.
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


def plot_loss_curves(result: dict) -> None:
    """Curvas de train_loss y val_loss por época para el modelo Transformer."""
    metrics = result['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_loss = metrics[['epoch', 'train_loss']].dropna()
    axes[0].plot(train_loss['epoch'], train_loss['train_loss'],
                 color='#1f77b4', linewidth=2, label='train_loss')
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Train Loss', fontsize=12)
    axes[0].set_title('Pérdida en Entrenamiento — Transformer', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    val_loss = metrics[['epoch', 'val_loss']].dropna()
    axes[1].plot(val_loss['epoch'], val_loss['val_loss'],
                 color='#ff7f0e', linewidth=2, label='val_loss')
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Pérdida en Validación — Transformer', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_FIGURES_DIR / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


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
