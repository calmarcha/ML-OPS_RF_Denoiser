"""
Configuración de loggers: CSV local y Weights & Biases.

Exporta:
    PersistentWandbLogger  — WandbLogger que no cierra el run al finalizar el
                             Trainer, permitiendo registrar métricas adicionales
                             (inferencia, test) tras el entrenamiento.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import os

import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger

from config import WANDB_PROJECT

# Carga variables de entorno desde el archivo .env en la raíz del proyecto
load_dotenv(Path(__file__).parent.parent / ".env")


class PersistentWandbLogger(WandbLogger):
    """WandbLogger que mantiene el run abierto después del Trainer.

    Por defecto, Lightning llama a wandb.finish() al terminar el entrenamiento.
    Esta subclase suprime ese comportamiento para poder seguir registrando
    métricas de inferencia y test en el mismo run tras el entrenamiento.
    El run debe cerrarse explícitamente llamando a
    ``result['wandb_logger'].experiment.finish()``.
    """

    def finalize(self, status: str) -> None:
        pass  # No cerrar el run aquí; se cerrará en el paso de guardado


def setup_wandb() -> None:
    """Realiza el login en W&B leyendo WANDB_API_KEY del entorno o del .env."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No se encontró WANDB_API_KEY.\n"
            "Añade la línea  WANDB_API_KEY=tu_clave  al archivo .env en la raíz del proyecto."
        )
    wandb.login(key=api_key)
    print(f"✓ W&B configurado — Proyecto: '{WANDB_PROJECT}'")
