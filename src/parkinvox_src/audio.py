"""
audio.py

EN: This module contains functions for audio processing.
TR: Bu modül ses işleme için fonksiyonlar içeriyor.
"""

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_audio(audio_path: str | Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    audio_data, sample_rate = sf.read(audio_path, always_2d=True, dtype="float64")
    # We did always_2d=True to ensure that audio is always loaded as a 2D array, even if it's mono. This is important for consistency in processing.

    number_of_samples = len(audio_data)

    """
    EN: Check if the audio file supplied is actually valid.
    TR: Sağlanan ses dosyasının gerçekten geçerli olup olmadığını kontrol edin.
    """

    if number_of_samples == 0:
        raise ValueError(f"[AUDIO_ERR] Audio file {audio_path} is empty.")

    has_invalid_values = not np.all(np.isfinite(audio_data))

    if has_invalid_values:
        raise ValueError(
            f"[AUDIO_ERR] Audio file {audio_path} contains invalid values."
        )

    number_of_channels = audio_data.shape[1]

    # Using the first channel.
    selected_channel = audio_data[:, 0]

    duration_seconds = len(selected_channel) / sample_rate

    metadata = {
        "source_sample_rate_hz": int(sample_rate),
        "source_channels": int(number_of_channels),
        "source_frames": int(len(selected_channel)),
        "duration_seconds": float(duration_seconds),
        "has_invalid_values": bool(has_invalid_values),
    }

    return selected_channel, int(sample_rate), metadata


def remove_dc_offset(audio_data: np.ndarray) -> np.ndarray:
    """
    EN: Remove DC offset from audio data.
    TR: Ses verilerinden DC ofsetini kaldır.
    """

    mean_value = np.mean(audio_data)
    centered_audio = audio_data - mean_value

    return centered_audio


def estimate_f0_values(audio_data: np.ndarray, sample_rate: int, minimum_f0_hz: float, maximum_f0_hz: float, frame_length: int, hop_length: int) -> np.ndarray:
    minimum_lag = max(1, int(sample_rate / maximum_f0_hz))
    maximum_lag = min(frame_length - 1, int(sample_rate / minimum_f0_hz))
    f0_values = np.full(max(1, 1 + (max(0, len(audio_data) - frame_length) // hop_length)), np.nan, dtype=float)

    for frame_index, start in enumerate(range(0, len(audio_data) - frame_length + 1, hop_length)):
        frame = audio_data[start:start + frame_length]
        frame = frame - np.mean(frame)
        frame *= np.hanning(frame_length)
        energy = np.dot(frame, frame)
        if energy <= np.finfo(float).eps:
            continue

        autocorrelation = np.correlate(frame, frame, mode="full")[frame_length - 1:]
        search_region = autocorrelation[minimum_lag:maximum_lag + 1]
        if len(search_region) == 0:
            continue

        peak_lag = minimum_lag + int(np.argmax(search_region))
        if autocorrelation[peak_lag] / autocorrelation[0] >= 0.1:
            f0_values[frame_index] = sample_rate / peak_lag

    return f0_values
