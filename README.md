# RF Denoiser - MLOps (Transformer)

**Almarcha Arias, G. Carlos** · Master en Deep Learning 2025–2026 · Asignatura MLOps

Proyecto de eliminación de ruido en la voz demodulada en radiocomunicaciones, mediante un modelo Transformer entrenado con diferentes ficheros de audio con voces y ruido rosa, aplicando buenas prácticas de MLOps.
---
## Enlaces a Github y Weights & Biases

https://github.com/calmarcha/ML-OPS_RF_Denoiser

https://wandb.ai/calmarcha-universidad-polit-cnica-de-madrid/RF-Denoiser/table?nw=nwusercalmarcha

---

## Descripción

El sistema procesa señales de audio en el **dominio de la frecuencia** (espectrogramas de magnitud). El modelo Transformer opera sobre los 257 bins de la STFT (Short Time Fourier Transform) y elimina el ruido aplicando atención multi-cabeza sobre la dimensión temporal.

- **Arquitectura**: Encoder Transformer con positional encoding
- **Entrada**: 257 bins STFT (N_FFT/2 + 1)
- **d_model**: 256
- **Cabezas de atención**: 4
- **Capas Transformer**: 4

El modelo se entrena con **PyTorch Lightning**, se monitoriza con **Weights & Biases** y toda **la configuración reside en un único fichero YAML**.

---

## Estructura del proyecto

```
Originalmente el proyecto estaba contenido en un único notebook, pero aplicando buenas prácticas de MLOps se ha refactorizado a una estructura modular. El código fuente se encuentra en `src/`, los tests en `tests/` y los resultados generados (CSVs, figuras) en `results/`. Los datos de entrenamiento, checkpoints, modelos exportados, logs y runs de W&B están excluidos del repositorio por contener ficheros binarios grandes o datos que se considera mejor no subir a GitHub.

``` 
 - config/
  - configuration.yaml   # Fuente única de verdad: hiperparámetros y rutas
- src/
  - config.py            # Carga configuration.yaml y expone constantes
  - data.py              # Carga de audio, espectrogramas y Dataset
  - logging_config.py    # PersistentWandbLogger + setup_wandb()
  - model_transformer.py # Arquitectura Transformer
  - train.py             # Función train_model() con loggers CSV + W&B
  - evaluate.py          # Inferencia y métricas de test
  - visualize.py         # Gráficas de pérdida y espectrogramas
  - export_models.py     # Exportación de pesos a models/
  - main.py              # Orquestador del pipeline completo
- tests/
  - test_data.py         # Tests de procesado de audio y AudioDenoisingDataset
- results/               # CSVs generados (métricas, tiempos, test)
- Dockerfile             # Imagen Docker con el pipeline completo
- .dockerignore          # Exclusiones del contexto de build
- pytest.ini             # Configuración de pytest (pythonpath = src)
- requirements.txt       # Requisitos del proyecto
- .env.example           # Clave de API de Weights & Biases

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

models:
  transformer:
    input_size: 257
    d_model: 256
    nhead: 4
    num_layers: 4
```

### 2. Clave API de W&B — `.env`

Copia la plantilla y añade tu clave:

```bash
cp .env.example .env
```

Edita `.env`:

```
WANDB_API_KEY=tu_clave_api_aquí
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
3. Entrenamiento del modelo Transformer
4. Análisis de tiempo de entrenamiento y curvas de pérdida
5. Medición de tiempo de inferencia
6. Evaluación en el dataset de test (audio no visto durante el entrenamiento)
7. Visualización de espectrogramas
8. Guardado de resultados en `results/` y cierre del run W&B

---

## Integración con Weights & Biases

El modelo Transformer genera un **run** en el proyecto `RF-Denoiser` de W&B con:

- **Config**: hiperparámetros completos del experimento
- **Métricas por época**: `train_loss`, `val_loss`
- **Resumen post-entrenamiento**: `training_time_s`, `best_val_loss`
- **Inferencia**: `inference_time_ms`, `realtime_ratio`
- **Test**: `test_l1_loss`, `test_mse`, `test_rmse`, `val_test_gap_pct`

El run queda disponible en:
`https://wandb.ai/calmarcha/RF-Denoiser`

Consulta [integración_W&B.txt](integración_W&B.txt) para la documentación completa de la integración.

---

## Integración con GitHub

El código fuente del proyecto está versionado en Git y publicado en GitHub, siguiendo las buenas prácticas de MLOps:

- **`.gitignore`**: excluye `.venv/`, `.env`, `training_data/`, `checkpoints/`, `models/`, `logs/` y `wandb/`.
- **`.env.example`**: plantilla versionada para la clave API de W&B; el fichero `.env` real nunca se versiona.
- **`requirements.txt`**: contrato reproducible de dependencias (`pip install -r requirements.txt`).
- **Convención de commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`).

Repositorio:
`https://github.com/calmarcha/ML-OPS_RF_Denoiser`

Consulta [integración_GitHub.txt](integración_GitHub.txt) para la documentación completa de la integración.

---

## Resultados

Los ficheros CSV generados en `results/` incluyen:

| Fichero | Contenido |
|---|---|
| `Transformer_metrics.csv` | Curvas de pérdida por época |
| `inference_times.csv` | Tiempo de inferencia y ratio de tiempo real |
| `test_results.csv` | Métricas de evaluación en test |

---

## Docker

La imagen incluye el modelo Transformer con los hiperparámetros que han obtenido el mejor resultado:

| Hiperparámetro | Valor |
|---|---|
| `sample_rate` | 16 000 Hz |
| `n_fft` | 512 |
| `hop_length` | 256 |
| `segment_length` | 2 s |
| `batch_size` | 8 |
| `learning_rate` | 0.0005 |
| `d_model` | 256 |
| `nhead` | 4 |
| `num_layers` | 4 |

**Métricas obtenidas** (`test_results.csv`): Test L1 Loss = 0.1201 · Test RMSE = 0.2688

### Construir la imagen

```bash
docker build -t rf-denoiser .
```

### Ejecutar el pipeline

```bash
docker run --rm \
  -v /ruta/local/training_data:/app/training_data \
  -v /ruta/local/results:/app/results \
  -e WANDB_API_KEY=tu_clave_api \
  rf-denoiser
```

> `training_data/` se monta desde el host porque los ficheros WAV son demasiado grandes para incluirlos en la imagen. Los resultados (CSVs y figuras) se persisten en el volumen `results/`.

---

## Tests

El proyecto incluye una suite de tests automatizados ejecutables con **pytest**.

```bash
# Desde la raíz del proyecto
pytest tests/ -v
```

| Fichero | Módulo bajo prueba | Nº tests |
|---|---|---|
| `tests/test_data.py` | `src/data.py` | 15 |

El fichero `pytest.ini` en la raíz del proyecto configura `pythonpath = src`, lo que permite a pytest resolver las importaciones sin necesidad de ficheros adicionales.

### `test_data.py`

- **`TestAudioToSpectrogram`** — forma de salida, bins de frecuencia, magnitud ≥ 0, fase en [−π, π], dtype float.
- **`TestSpectrogramToAudio`** — reconstrucción produce array 1-D, longitud cercana a la original, valores finitos.
- **`TestCreateSegments`** — número de segmentos, longitud exacta, segmento ruidoso ≠ limpio.
- **`TestAudioDenoisingDataset`** — longitud del dataset, tipos de tensores, forma `[1, F, T]`, dtype float32.

---

## Licencia

Proyecto de MLOPS - Master en Deep Learning - Mayo 2026.
