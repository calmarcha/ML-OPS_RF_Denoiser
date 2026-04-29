"""
Exporta el mejor checkpoint (.ckpt) de cada arquitectura como archivo .pth
(state_dict de PyTorch) en la carpeta models/.

Uso:
    cd <raíz del proyecto>
    python src/export_models.py
"""

import re
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch

from model_GRU         import GRUDenoiser
from model_LSTM        import LSTMDenoiser
from model_CRN         import CRNDenoiser
from model_transformer import TransformerDenoiser

# Mapeo: nombre → clase Lightning
MODEL_CLASSES = {
    "GRU":         GRUDenoiser,
    "LSTM":        LSTMDenoiser,
    "CRN":         CRNDenoiser,
    "Transformer": TransformerDenoiser,
}

CHECKPOINTS_DIR = _ROOT_DIR / "checkpoints"
MODELS_DIR      = _ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def _val_loss_from_filename(path: Path) -> float:
    """Extrae el val_loss del nombre de fichero del checkpoint."""
    match = re.search(r"val_loss=([\d.]+)", path.stem)
    return float(match.group(1)) if match else float("inf")


def _best_checkpoint(model_name: str) -> Path:
    """Devuelve el checkpoint con menor val_loss para un modelo dado."""
    ckpt_dir = CHECKPOINTS_DIR / model_name
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No se encontraron checkpoints en {ckpt_dir}")
    return min(checkpoints, key=_val_loss_from_filename)


def export_model(model_name: str, model_cls) -> None:
    best_ckpt = _best_checkpoint(model_name)
    print(f"[{model_name}] Cargando: {best_ckpt.name}")

    model = model_cls.load_from_checkpoint(str(best_ckpt), map_location="cpu")
    model.eval()

    out_path = MODELS_DIR / f"{model_name}_best.pth"
    torch.save(model.state_dict(), out_path)
    print(f"[{model_name}] Guardado: {out_path.relative_to(_ROOT_DIR)}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPORTANDO MODELOS A models/")
    print("=" * 60)

    for name, cls in MODEL_CLASSES.items():
        export_model(name, cls)

    print("\nExportación completada.")
    print(f"Archivos generados en: {MODELS_DIR.relative_to(_ROOT_DIR)}/")
