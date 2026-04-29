"""
Arquitectura GRU para denoising de audio en el dominio frecuencial.

Clase pública:
    GRUDenoiser — LightningModule con GRU bidireccional (2 capas, 128 unidades)
                  y capa FC, pérdida L1Loss, optimizador Adam.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn
import lightning as L

from config import LEARNING_RATE


class GRUDenoiser(L.LightningModule):
    """Modelo basado en GRU para audio denoising."""

    def __init__(
        self,
        input_size: int = 257,
        hidden_size: int = 128,
        num_layers: int = 2,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, input_size)
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, freq, time]
        batch, _, freq, time = x.shape
        x = x.squeeze(1).permute(0, 2, 1)   # [batch, time, freq]
        out, _ = self.gru(x)
        out = self.fc(out)
        out = out.permute(0, 2, 1).unsqueeze(1)  # [batch, 1, freq, time]
        return out

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        loss = self.criterion(self(noisy), clean)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        loss = self.criterion(self(noisy), clean)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
