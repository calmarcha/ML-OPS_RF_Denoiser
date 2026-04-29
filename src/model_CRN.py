"""
Arquitectura CRN (Convolutional Recurrent Network) para denoising de audio.

Clase pública:
    CRNDenoiser — LightningModule con encoder CNN (3 bloques Conv2d),
                  LSTM bidireccional y decoder CNN con skip connections.
                  Pérdida L1Loss, optimizador Adam.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from config import LEARNING_RATE


class CRNDenoiser(L.LightningModule):
    """Modelo CRN (Convolutional Recurrent Network) para audio denoising."""

    def __init__(self, learning_rate: float = LEARNING_RATE) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )

        # ── Proyección de entrada al LSTM ─────────────────────────────────────
        # Después de enc3: [batch, 64, 32, 15] → 64 * 32 = 2048 features
        self.lstm_input_size = 128
        self.lstm_projection = nn.Linear(2048, self.lstm_input_size)

        # ── LSTM bidireccional ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Proyección de salida del LSTM (256 = 128 * 2 bidireccional)
        self.lstm_output_proj = nn.Linear(256, 2048)

        # ── Decoder con skip connections ──────────────────────────────────────
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # 64 = 32 + 32 (skip)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),   # 32 = 16 + 16 (skip)
            nn.Sigmoid(),
        )

        self.criterion = nn.L1Loss()

    # ── Utilidad de alineación de dimensiones ─────────────────────────────────
    @staticmethod
    def _match_dimensions(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ajusta H y W de x para que coincidan con target mediante padding/crop."""
        diff_h = target.shape[2] - x.shape[2]
        diff_w = target.shape[3] - x.shape[3]
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, (0, max(0, diff_w), 0, max(0, diff_h)))
        if diff_h < 0 or diff_w < 0:
            x = x[:, :, : target.shape[2], : target.shape[3]]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # ── LSTM ─────────────────────────────────────────────────────────────
        batch, channels, freq, time = e3.shape
        lstm_in = e3.permute(0, 3, 1, 2).contiguous().view(batch, time, -1)
        lstm_in = self.lstm_projection(lstm_in)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_output_proj(lstm_out)
        lstm_out = lstm_out.view(batch, time, channels, freq).permute(0, 2, 3, 1).contiguous()

        # ── Decoder ──────────────────────────────────────────────────────────
        d3 = self._match_dimensions(self.dec3(lstm_out), e2)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self._match_dimensions(self.dec2(d3), e1)
        d2 = torch.cat([d2, e1], dim=1)

        out = self._match_dimensions(self.dec1(d2), x)
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
