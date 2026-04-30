# RF Denoiser — MLOps

**Almarcha Arias, G. Carlos** · Master en Deep Learning 2025–2026 · Asignatura MLOps

Proyecto de eliminación de ruido en señales de radio frecuencia (RF) mediante cuatro arquitecturas de red neuronal entrenadas y comparadas con un pipeline MLOps completo.

---

## Descripción

El sistema procesa señales de audio en el **dominio frecuencial** (espectrogramas de magnitud). Cuatro modelos compiten bajo las mismas condiciones de entrenamiento, validación y test:

| Modelo | Arquitectura | Parámetros clave |
|---|---|---|
| **GRU** | GRU bidireccional × 2 capas, 128 unidades | entrada 257 bins STFT |
| **LSTM** | LSTM bidireccional × 2 capas, 128 unidades | entrada 257 bins STFT |
| **CRN** | Convolutional Recurrent Network (encoder-GRU-decoder) | skip connections |
| **Transformer** | Encoder Transformer con positional encoding | 4 cabezas de atención |

Todos los modelos se entrenan con **PyTorch Lightning**, se monitorizan con **Weights & Biases** y toda su configuración reside en un único fichero YAML.

---

## Estructura del proyecto

```
.
├── config/
│   └── configuration.yaml   # Fuente única de verdad: hiperparámetros y rutas
├── src/
│   ├── config.py            # Carga configuration.yaml y expone constantes
│   ├── data.py              # Carga de audio, espectrogramas y Dataset
│   ├── logging_config.py    # PersistentWandbLogger + setup_wandb()
│   ├── model_GRU.py         # Arquitectura GRU
│   ├── model_LSTM.py        # Arquitectura LSTM
│   ├── model_CRN.py         # Arquitectura CRN
│   ├── model_transformer.py # Arquitectura Transformer
│   ├── train.py             # Función train_model() con loggers CSV + W&B
│   ├── evaluate.py          # Inferencia y métricas de test
│   ├── visualize.py         # Gráficas de pérdida, espectrogramas e inferencia
│   ├── export_models.py     # Exportación de pesos a models/
│   └── main.py              # Orquestador del pipeline completo
├── results/                 # CSVs generados (métricas, tiempos, resumen)
├── requirements.txt
├── .env.example             # Plantilla para la clave API de W&B
└── integración_W&B.txt      # Documentación detallada de la integración W&B
```

> Los directorios `training_data/`, `checkpoints/`, `models/`, `logs/` y `wandb/` están excluidos del repositorio por contener ficheros binarios grandes o datos sensibles.

---

## Requisitos

- Python 3.10+
- CUDA 11.8+ recomendado (el código funciona también en CPU)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/calmarcha/ML-OPS_RF_Denoiser.git
cd ML-OPS_RF_Denoiser

# 2. Crear y activar entorno virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración

### 1. Hiperparámetros — `config/configuration.yaml`

Todos los parámetros del experimento se gestionan desde este fichero:

```yaml
audio:
  sample_rate: 16000
  n_fft: 512
  hop_length: 256
  segment_length: 2       # segundos por segmento

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.0005

wandb:
  project: RF-Denoiser
```

### 2. Clave API de W&B — `.env`

Copia la plantilla y añade tu clave:

```bash
cp .env.example .env
```

Edita `.env`:

```
WANDB_API_KEY=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
```

Obtén tu clave en [wandb.ai/settings](https://wandb.ai/settings).

### 3. Datos de entrenamiento

Coloca los ficheros de audio en `training_data/` con los nombres indicados en `config/configuration.yaml` (sección `data`).

---

## Uso

Ejecuta el pipeline completo desde la raíz del proyecto:

```bash
python src/main.py
```

El script ejecuta en orden:

1. Login en W&B y detección de dispositivo (CPU / GPU)
2. Carga y segmentación de los audios de entrenamiento
3. Entrenamiento de GRU, LSTM, CRN y Transformer
4. Análisis de tiempos de entrenamiento y curvas de pérdida
5. Medición de tiempos de inferencia
6. Evaluación en el dataset de test (audio no visto durante el entrenamiento)
7. Visualización de espectrogramas con el mejor modelo
8. Guardado de resultados en `results/` y cierre de runs W&B

---

## Integración con Weights & Biases

Cada modelo genera un **run independiente** en el proyecto `RF-Denoiser` de W&B con:

- **Config**: hiperparámetros completos del experimento
- **Métricas por época**: `train_loss`, `val_loss`
- **Resumen post-entrenamiento**: `training_time_s`, `best_val_loss`
- **Inferencia**: `inference_time_ms`, `realtime_ratio`
- **Test**: `test_l1_loss`, `test_mse`, `test_rmse`, `val_test_gap_pct`
- **Tabla comparativa**: `comparison_table` con el resumen de todos los modelos

Los runs quedan disponibles en:
`https://wandb.ai/calmarcha/RF-Denoiser`

Consulta [integración_W&B.txt](integración_W&B.txt) para la documentación completa de la integración.

---

## Resultados

Los ficheros CSV generados en `results/` incluyen:

| Fichero | Contenido |
|---|---|
| `model_comparison_summary.csv` | Resumen comparativo de los cuatro modelos |
| `inference_times.csv` | Tiempos de inferencia por modelo |
| `test_results.csv` | Métricas de evaluación en test |
| `GRU_metrics.csv` / `LSTM_metrics.csv` / ... | Curvas de pérdida por modelo |

---

## Licencia

Proyecto académico — Master en Deep Learning, Abril 2026.
