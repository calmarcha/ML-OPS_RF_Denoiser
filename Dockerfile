# =============================================================================
# RF Denoiser — Dockerfile
# Modelo: Transformer  |  Mejores hiperparámetros registrados en test_results.csv
#   · Test L1 Loss : 0.1201
#   · Test RMSE    : 0.2688
#
# Hiperparámetros fijados en config/configuration.yaml:
#   audio    → sample_rate=16000, n_fft=512, hop_length=256, segment_length=2s
#   training → batch_size=8, epochs=100, learning_rate=0.0005
#   model    → d_model=256, nhead=4, num_layers=4, input_size=257
# =============================================================================

FROM python:3.11-slim

# ── Dependencias del sistema requeridas por librosa / soundfile ───────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Código fuente y configuración ─────────────────────────────────────────────
COPY config/  config/
COPY src/     src/
COPY models/  models/

# ── Datos de entrenamiento: montar en tiempo de ejecución ─────────────────────
# docker run -v /ruta/local/training_data:/app/training_data ...
VOLUME ["/app/training_data"]

# ── Resultados: montar para persistir los CSVs y figuras generados ────────────
# docker run -v /ruta/local/results:/app/results ...
VOLUME ["/app/results"]

# ── Clave API de Weights & Biases (pasar con -e WANDB_API_KEY=...) ───────────
ENV WANDB_API_KEY=""

# ── Evitar buffers en stdout para ver logs en tiempo real ────────────────────
ENV PYTHONUNBUFFERED=1

# ── Punto de entrada ──────────────────────────────────────────────────────────
CMD ["python", "src/main.py"]
