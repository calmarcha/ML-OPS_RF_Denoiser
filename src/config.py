"""
Configuración global de hiperparámetros y rutas del proyecto RF Denoiser.

Los valores se cargan desde config/configuration.yaml, buscando el fichero
en la raíz del proyecto (dos niveles por encima de este script).
"""

from pathlib import Path
import yaml
import torch

# ── Carga del fichero YAML ────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "configuration.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ── Hiperparámetros de audio ──────────────────────────────────────────────────
SAMPLE_RATE    = _cfg["audio"]["sample_rate"]
N_FFT          = _cfg["audio"]["n_fft"]
HOP_LENGTH     = _cfg["audio"]["hop_length"]
SEGMENT_LENGTH = _cfg["audio"]["segment_length"]

# ── Hiperparámetros de entrenamiento ─────────────────────────────────────────
BATCH_SIZE    = _cfg["training"]["batch_size"]
EPOCHS        = _cfg["training"]["epochs"]
LEARNING_RATE = _cfg["training"]["learning_rate"]

# ── Rutas de archivos de entrenamiento ───────────────────────────────────────
NOISE_FILE = _cfg["data"]["train"]["noise_file"]
VOICE_FILE = _cfg["data"]["train"]["voice_file"]

# ── Rutas de archivos de test ─────────────────────────────────────────────────
TEST_NOISE_FILE = _cfg["data"]["test"]["noise_file"]
TEST_VOICE_FILE = _cfg["data"]["test"]["voice_file"]

# ── Directorios de salida ─────────────────────────────────────────────────────
CHECKPOINTS_DIR = _cfg["paths"]["checkpoints_dir"]
LOGS_DIR        = _cfg["paths"]["logs_dir"]
RESULTS_DIR     = _cfg["paths"]["results_dir"]
FIGURES_DIR     = _cfg["paths"]["figures_dir"]

# ── W&B ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = _cfg["wandb"]["project"]

# ── Dispositivo ───────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
