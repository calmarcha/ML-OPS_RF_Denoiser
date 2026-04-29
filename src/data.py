"""
Carga, preprocesado de audio y construcción del Dataset de PyTorch.

Funciones públicas:
    load_audio          — Carga y normaliza un archivo WAV.
    audio_to_spectrogram — Audio → espectrograma de magnitud + fase.
    spectrogram_to_audio — Espectrograma (magnitud + fase) → audio.
    create_segments     — Crea pares (limpio, ruidoso) de segmentos de audio.

Clases públicas:
    AudioDenoisingDataset — Dataset de PyTorch para audio denoising.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, SEGMENT_LENGTH


# ── Utilidades de audio ───────────────────────────────────────────────────────

def load_audio(filepath: str) -> np.ndarray:
    """Carga un archivo WAV, resamplea a SAMPLE_RATE y normaliza la amplitud."""
    audio, sr = sf.read(filepath)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio


def audio_to_spectrogram(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte audio a espectrograma de magnitud y fase."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase


def spectrogram_to_audio(
    magnitude: np.ndarray,
    phase: np.ndarray,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Convierte espectrograma (magnitud + fase) a audio."""
    stft = magnitude * np.exp(1j * phase)
    return librosa.istft(stft, hop_length=hop_length)


def create_segments(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    segment_samples: int,
    noise_level: float = 0.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Divide los audios en segmentos y mezcla voz limpia con ruido.

    Args:
        clean_audio:     Audio de voz limpia (normalizado).
        noise_audio:     Audio de ruido puro (normalizado).
        segment_samples: Número de muestras por segmento.
        noise_level:     Factor de mezcla del ruido (0–1).

    Returns:
        Tupla (segments_clean, segments_noisy).
    """
    segments_clean: list[np.ndarray] = []
    segments_noisy: list[np.ndarray] = []

    min_length = min(len(clean_audio), len(noise_audio))
    clean_audio = clean_audio[:min_length]
    noise_audio = noise_audio[:min_length]

    num_segments = min_length // segment_samples
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        clean_seg = clean_audio[start:end]
        noise_seg = noise_audio[start:end]
        noisy_seg = clean_seg + noise_level * noise_seg
        segments_clean.append(clean_seg)
        segments_noisy.append(noisy_seg)

    return segments_clean, segments_noisy


# ── Dataset ───────────────────────────────────────────────────────────────────

class AudioDenoisingDataset(Dataset):
    """Dataset de PyTorch para audio denoising en el dominio frecuencial."""

    def __init__(
        self,
        clean_segments: list[np.ndarray],
        noisy_segments: list[np.ndarray],
    ) -> None:
        self.clean_spectrograms: list[np.ndarray] = []
        self.noisy_spectrograms: list[np.ndarray] = []

        print("Convirtiendo segmentos a espectrogramas...")
        for clean, noisy in zip(clean_segments, noisy_segments):
            clean_mag, _ = audio_to_spectrogram(clean)
            noisy_mag, _ = audio_to_spectrogram(noisy)
            self.clean_spectrograms.append(clean_mag)
            self.noisy_spectrograms.append(noisy_mag)

    def __len__(self) -> int:
        return len(self.clean_spectrograms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = torch.FloatTensor(self.clean_spectrograms[idx]).unsqueeze(0)  # [1, freq, time]
        noisy = torch.FloatTensor(self.noisy_spectrograms[idx]).unsqueeze(0)
        return noisy, clean


# ── Función de conveniencia ───────────────────────────────────────────────────

def build_datasets(
    voice_file: str,
    noise_file: str,
    val_split: float = 0.2,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Carga los audios, crea segmentos y devuelve (train_dataset, val_dataset)."""
    print("Cargando archivos de audio...")
    noise_audio = load_audio(noise_file)
    voice_audio = load_audio(voice_file)
    print(f"  Ruido shape: {noise_audio.shape} | Voz shape: {voice_audio.shape}")

    segment_samples = int(SEGMENT_LENGTH * SAMPLE_RATE)
    print("Creando segmentos de entrenamiento...")
    clean_segs, noisy_segs = create_segments(voice_audio, noise_audio, segment_samples)
    print(f"  Total segmentos: {len(clean_segs)}")

    dataset = AudioDenoisingDataset(clean_segs, noisy_segs)
    print(f"  Dataset creado con {len(dataset)} muestras")
    print(f"  Forma espectrograma: {dataset.noisy_spectrograms[0].shape}")

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


def build_test_dataset(
    voice_file: str,
    noise_file: str,
) -> tuple[AudioDenoisingDataset, int]:
    """Carga los audios de test y devuelve (test_dataset, segment_samples)."""
    print("Cargando archivos de audio de TEST...")
    noise_audio = load_audio(noise_file)
    voice_audio = load_audio(voice_file)

    segment_samples = int(SEGMENT_LENGTH * SAMPLE_RATE)
    clean_segs, noisy_segs = create_segments(voice_audio, noise_audio, segment_samples)
    print(f"  Segmentos de test creados: {len(clean_segs)}")

    test_dataset = AudioDenoisingDataset(clean_segs, noisy_segs)
    print(f"  Test dataset: {len(test_dataset)} muestras")
    return test_dataset, segment_samples
