"""
Arquitectura Transformer para denoising de audio en el dominio frecuencial.

Clase pública:
    TransformerDenoiser — LightningModule con proyección de entrada,
                          Positional Encoding aprendido, TransformerEncoder
                          (4 capas, d_model=256, nhead=4) y proyección de salida.
                          Pérdida L1Loss, optimizador Adam.
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


class TransformerDenoiser(L.LightningModule):
    """Modelo basado en Transformer para audio denoising."""

    def __init__(
        self,
        input_size: int = 257,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_size)

        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, freq, time]
        batch, _, freq, time = x.shape
        x = x.squeeze(1).permute(0, 2, 1)           # [batch, time, freq]
        x = self.input_proj(x)                        # [batch, time, d_model]
        x = x + self.pos_encoding[:, :time, :]
        x = self.transformer(x)
        x = self.output_proj(x)                       # [batch, time, freq]
        x = x.permute(0, 2, 1).unsqueeze(1)          # [batch, 1, freq, time]
        return x

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
