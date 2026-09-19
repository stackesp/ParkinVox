"""
prosody.py

EN: Prosody, fluency, pause, pitch and intensity features.
TR: Prozodi, akıcılık, duraklama, perde ve şiddet özellikleri.
"""

from typing import Any

import librosa
import numpy as np

from parkinvox_src.audio import estimate_f0_values
from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    if len(mask) == 0:
        return np.array([], dtype=int)

    padded_mask = np.pad(mask.astype(int), (1, 1))
    changes = np.diff(padded_mask)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return ends - starts


def calculate_prosody_features(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> dict[str, Any]:
    rms_values = librosa.feature.rms(y=audio, frame_length=config.frame_length, hop_length=config.hop_length,)[0]

    reference_amplitude = max(float(np.max(rms_values)), EPSILON)
    rms_decibels = librosa.amplitude_to_db(rms_values + EPSILON, ref=reference_amplitude)
    speech_mask = rms_decibels > -config.silence_threshold_db

    frame_seconds = config.hop_length / sample_rate
    pause_lengths = _run_lengths(~speech_mask) * frame_seconds
    valid_pause_lengths = pause_lengths[pause_lengths >= 0.15]

    f0_values = estimate_f0_values(audio, sample_rate, config.minimum_f0_hz, config.maximum_f0_hz, config.frame_length, config.hop_length)
    valid_f0_values = f0_values[np.isfinite(f0_values)]

    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    threshold = np.median(onset_envelope) + np.std(onset_envelope)
    onset_frames = np.flatnonzero(
        (onset_envelope[1:-1] > onset_envelope[:-2]) & (onset_envelope[1:-1] >= onset_envelope[2:]) & (onset_envelope[1:-1] > threshold)
    ) + 1
    estimated_syllable_count = len(onset_frames)

    duration_seconds = len(audio) / sample_rate
    speech_seconds = float(np.sum(speech_mask) * frame_seconds)
    pause_seconds = float(np.sum(valid_pause_lengths))

    return {
        "pros_duration_seconds": float(duration_seconds),
        "pros_speech_seconds": speech_seconds,
        "pros_voiced_frame_fraction": float(np.mean(speech_mask)),
        "pros_pause_count": int(len(valid_pause_lengths)),
        "pros_pause_total_seconds": pause_seconds,
        "pros_pause_ratio": float(pause_seconds / max(duration_seconds, EPSILON)),
        "pros_pause_mean_seconds": float(np.mean(valid_pause_lengths)) if len(valid_pause_lengths) else 0.0,
        "pros_pause_std_seconds": float(np.std(valid_pause_lengths)) if len(valid_pause_lengths) else 0.0,
        "pros_f0_range_hz": float(np.ptp(valid_f0_values)) if len(valid_f0_values) else 0.0,
        "pros_f0_iqr_hz": float(np.subtract(*np.percentile(valid_f0_values, [75, 25]))) if len(valid_f0_values) else 0.0,
        "pros_intensity_range_db": float(np.ptp(rms_decibels)),
        "pros_intensity_std_db": float(np.std(rms_decibels)),
        "pros_estimated_syllable_count": int(estimated_syllable_count),
        "pros_estimated_speech_rate": float(estimated_syllable_count / max(duration_seconds, EPSILON)),
        "pros_estimated_articulation_rate": float(estimated_syllable_count / max(speech_seconds, EPSILON)),
    }
