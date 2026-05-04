"""
Tests para las funciones de procesado de audio y el Dataset definidos en src/data.py.

Cubre:
    - audio_to_spectrogram : formas de salida y tipos de datos.
    - spectrogram_to_audio : reconstrucción audio → espectrograma → audio.
    - create_segments       : número y forma de los segmentos generados.
    - AudioDenoisingDataset : longitud del dataset y forma de los tensores.
"""

import numpy as np
import pytest
import torch

from data import (
    AudioDenoisingDataset,
    audio_to_spectrogram,
    create_segments,
    spectrogram_to_audio,
)
from config import N_FFT, HOP_LENGTH, SAMPLE_RATE

# ── Fixtures ──────────────────────────────────────────────────────────────────

SEGMENT_SAMPLES = SAMPLE_RATE * 2  # 2 segundos de audio sintético


@pytest.fixture
def synthetic_audio() -> np.ndarray:
    """Señal senoidal sintética normalizada de 2 segundos."""
    t = np.linspace(0, 2, SEGMENT_SAMPLES, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio / (np.max(np.abs(audio)) + 1e-8)


@pytest.fixture
def synthetic_noise() -> np.ndarray:
    """Ruido blanco sintético normalizado de 2 segundos."""
    rng = np.random.default_rng(seed=42)
    noise = rng.standard_normal(SEGMENT_SAMPLES).astype(np.float32)
    return noise / (np.max(np.abs(noise)) + 1e-8)


# ── Tests de audio_to_spectrogram ─────────────────────────────────────────────

class TestAudioToSpectrogram:
    """Verifica que audio_to_spectrogram produce salidas con forma y tipo correctos."""

    def test_output_shapes_are_consistent(self, synthetic_audio):
        magnitude, phase = audio_to_spectrogram(synthetic_audio)
        assert magnitude.shape == phase.shape, (
            "magnitude y phase deben tener la misma forma"
        )

    def test_frequency_bins(self, synthetic_audio):
        """El número de bins de frecuencia debe ser N_FFT // 2 + 1."""
        magnitude, _ = audio_to_spectrogram(synthetic_audio)
        expected_freq_bins = N_FFT // 2 + 1
        assert magnitude.shape[0] == expected_freq_bins, (
            f"Se esperaban {expected_freq_bins} bins de frecuencia, "
            f"se obtuvieron {magnitude.shape[0]}"
        )

    def test_magnitude_is_non_negative(self, synthetic_audio):
        magnitude, _ = audio_to_spectrogram(synthetic_audio)
        assert np.all(magnitude >= 0), "La magnitud del espectrograma debe ser ≥ 0"

    def test_phase_range(self, synthetic_audio):
        """La fase debe estar en el rango [-π, π]."""
        _, phase = audio_to_spectrogram(synthetic_audio)
        assert np.all(phase >= -np.pi) and np.all(phase <= np.pi), (
            "La fase debe estar en [-π, π]"
        )

    def test_output_dtype_is_float(self, synthetic_audio):
        magnitude, phase = audio_to_spectrogram(synthetic_audio)
        assert np.issubdtype(magnitude.dtype, np.floating)
        assert np.issubdtype(phase.dtype, np.floating)


# ── Tests de spectrogram_to_audio ─────────────────────────────────────────────

class TestSpectrogramToAudio:
    """Verifica la reconstrucción espectrograma → audio."""

    def test_roundtrip_produces_1d_array(self, synthetic_audio):
        magnitude, phase = audio_to_spectrogram(synthetic_audio)
        reconstructed = spectrogram_to_audio(magnitude, phase)
        assert reconstructed.ndim == 1, "El audio reconstruido debe ser un array 1-D"

    def test_roundtrip_length_is_close_to_original(self, synthetic_audio):
        """La longitud reconstruida debe estar cerca de la original (±hop_length)."""
        magnitude, phase = audio_to_spectrogram(synthetic_audio)
        reconstructed = spectrogram_to_audio(magnitude, phase)
        diff = abs(len(reconstructed) - len(synthetic_audio))
        assert diff <= HOP_LENGTH, (
            f"Diferencia de longitud {diff} supera la tolerancia de {HOP_LENGTH}"
        )

    def test_roundtrip_values_are_finite(self, synthetic_audio):
        magnitude, phase = audio_to_spectrogram(synthetic_audio)
        reconstructed = spectrogram_to_audio(magnitude, phase)
        assert np.all(np.isfinite(reconstructed)), (
            "El audio reconstruido no debe contener NaN ni Inf"
        )


# ── Tests de create_segments ──────────────────────────────────────────────────

class TestCreateSegments:
    """Verifica que create_segments genera segmentos con la forma correcta."""

    def test_segment_count(self, synthetic_audio, synthetic_noise):
        duration_samples = len(synthetic_audio)
        segments_clean, segments_noisy = create_segments(
            synthetic_audio, synthetic_noise, segment_samples=duration_samples
        )
        assert len(segments_clean) == len(segments_noisy), (
            "El número de segmentos limpios y ruidosos debe coincidir"
        )
        assert len(segments_clean) >= 1, "Debe generarse al menos un segmento"

    def test_segment_length(self, synthetic_audio, synthetic_noise):
        seg_len = SAMPLE_RATE  # 1 segundo
        segments_clean, segments_noisy = create_segments(
            synthetic_audio, synthetic_noise, segment_samples=seg_len
        )
        for seg in segments_clean + segments_noisy:
            assert len(seg) == seg_len, (
                f"Longitud de segmento esperada {seg_len}, obtenida {len(seg)}"
            )

    def test_noisy_differs_from_clean(self, synthetic_audio, synthetic_noise):
        seg_len = SAMPLE_RATE
        segments_clean, segments_noisy = create_segments(
            synthetic_audio, synthetic_noise, segment_samples=seg_len
        )
        # Los segmentos ruidosos no deben ser idénticos a los limpios
        assert not np.allclose(segments_clean[0], segments_noisy[0]), (
            "El segmento ruidoso no debe ser idéntico al limpio"
        )


# ── Tests de AudioDenoisingDataset ────────────────────────────────────────────

class TestAudioDenoisingDataset:
    """Verifica el comportamiento del Dataset de PyTorch."""

    @pytest.fixture
    def dataset(self, synthetic_audio, synthetic_noise):
        seg_len = SAMPLE_RATE
        clean_segs, noisy_segs = create_segments(
            synthetic_audio, synthetic_noise, segment_samples=seg_len
        )
        return AudioDenoisingDataset(clean_segs, noisy_segs)

    def test_dataset_length(self, dataset, synthetic_audio):
        expected_len = len(synthetic_audio) // SAMPLE_RATE
        assert len(dataset) == expected_len

    def test_getitem_returns_two_tensors(self, dataset):
        noisy, clean = dataset[0]
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)

    def test_tensor_shape_has_channel_dim(self, dataset):
        """Los tensores deben tener forma [1, freq_bins, time_frames]."""
        noisy, clean = dataset[0]
        assert noisy.ndim == 3, "El tensor debe ser 3-D: [C, F, T]"
        assert noisy.shape[0] == 1, "El canal debe ser 1"
        assert noisy.shape == clean.shape, (
            "noisy y clean deben tener la misma forma"
        )

    def test_tensor_dtype_is_float32(self, dataset):
        noisy, clean = dataset[0]
        assert noisy.dtype == torch.float32
        assert clean.dtype == torch.float32
