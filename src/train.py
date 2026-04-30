"""
Función de entrenamiento de un único modelo con Lightning Trainer.

Función pública:
    train_model — Configura DataLoaders, callbacks, loggers CSV + W&B
                  y lanza el Trainer. Devuelve un dict con modelo, métricas,
                  tiempos y logger de W&B.
"""

import os
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    N_FFT,
    HOP_LENGTH,
    SAMPLE_RATE,
    SEGMENT_LENGTH,
    WANDB_PROJECT,
    device,
)
from logging_config import PersistentWandbLogger


def train_model(
    model: L.LightningModule,
    model_name: str,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    epochs: int = EPOCHS,
) -> dict:
    """Entrena un modelo y retorna las métricas y artefactos.

    Args:
        model:         Instancia del modelo LightningModule a entrenar.
        model_name:    Nombre del modelo (usado en rutas y logs).
        train_dataset: Dataset de entrenamiento.
        val_dataset:   Dataset de validación.
        epochs:        Número máximo de épocas.

    Returns:
        Dict con claves:
            model, trainer, metrics (DataFrame), training_time,
            best_checkpoint, logger_version, wandb_logger.
    """
    print(f"\n{'='*60}")
    print(f"Entrenando {model_name}")
    print(f"{'='*60}\n")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=use_pin_memory,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{model_name}',
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
    )

    # ── Loggers ───────────────────────────────────────────────────────────────
    csv_logger = CSVLogger('logs', name=model_name)
    wandb_logger = PersistentWandbLogger(
        project=WANDB_PROJECT,
        name=model_name,
        log_model=False,
        reinit=True,
        config={
            'model':          model_name,
            'epochs':         epochs,
            'batch_size':     BATCH_SIZE,
            'learning_rate':  LEARNING_RATE,
            'sample_rate':    SAMPLE_RATE,
            'n_fft':          N_FFT,
            'hop_length':     HOP_LENGTH,
            'segment_length': SEGMENT_LENGTH,
            'device':         str(device),
        },
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[csv_logger, wandb_logger],
        accelerator='auto',
        devices=1,
        log_every_n_steps=5,
        enable_progress_bar=False,
    )

    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = time.time() - start_time

    print(f"\n{model_name} entrenado en {training_time:.2f} segundos")

    # ── Cargar métricas CSV ───────────────────────────────────────────────────
    logger_version = csv_logger.version
    metrics_file = f'logs/{model_name}/version_{logger_version}/metrics.csv'

    wait_time = 0
    while not os.path.exists(metrics_file) and wait_time < 10:
        time.sleep(0.5)
        wait_time += 0.5

    if os.path.exists(metrics_file):
        metrics_df = pd.read_csv(metrics_file)
    else:
        print(f"Advertencia: No se encontró {metrics_file}")
        metrics_df = pd.DataFrame()

    # ── Registrar métricas de entrenamiento en W&B ────────────────────────────
    best_val_loss = checkpoint_callback.best_model_score
    wandb_logger.experiment.log({
        'training_time_s': training_time,
        'best_val_loss': best_val_loss.item() if best_val_loss is not None else None,
    })

    return {
        'model':           model,
        'trainer':         trainer,
        'metrics':         metrics_df,
        'training_time':   training_time,
        'best_checkpoint': checkpoint_callback.best_model_path,
        'logger_version':  logger_version,
        'wandb_logger':    wandb_logger,
    }
