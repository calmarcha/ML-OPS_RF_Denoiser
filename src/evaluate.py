"""
Evaluación de modelos: tiempo de inferencia y métricas en dataset de test.

Funciones públicas:
    measure_inference_time — Tiempo promedio de inferencia en ms (100 iteraciones).
    evaluate_on_test        — Calcula Test L1 Loss, MSE y RMSE sobre un DataLoader.
"""

import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from config import device, BATCH_SIZE


def measure_inference_time(
    model: L.LightningModule,
    num_iterations: int = 100,
) -> float:
    """Mide el tiempo de inferencia promedio del modelo.

    Se realizan 10 iteraciones de warm-up antes de la medición.

    Args:
        model:           Modelo LightningModule a evaluar.
        num_iterations:  Número de iteraciones para promediar.

    Returns:
        Tiempo promedio de inferencia en milisegundos.
    """
    model.eval()
    model = model.to(device)

    test_input = torch.randn(1, 1, 257, 126).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(test_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(test_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    avg_ms = (time.time() - start) / num_iterations * 1000
    return avg_ms


def evaluate_on_test(
    model: L.LightningModule,
    test_dataset: torch.utils.data.Dataset,
) -> dict:
    """Evalúa un modelo sobre el dataset de test.

    Args:
        model:        Modelo LightningModule ya entrenado.
        test_dataset: Dataset de test (AudioDenoisingDataset).

    Returns:
        Dict con claves: test_loss (L1), test_mse, test_rmse.
    """
    model.eval()
    model = model.to(device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    l1_losses: list[float] = []
    mse_losses: list[float] = []

    with torch.no_grad():
        for noisy_batch, clean_batch in test_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            output = model(noisy_batch)
            l1_losses.append(nn.L1Loss()(output, clean_batch).item())
            mse_losses.append(nn.MSELoss()(output, clean_batch).item())

    avg_l1  = float(np.mean(l1_losses))
    avg_mse = float(np.mean(mse_losses))
    return {
        'test_loss': avg_l1,
        'test_mse':  avg_mse,
        'test_rmse': float(np.sqrt(avg_mse)),
    }
