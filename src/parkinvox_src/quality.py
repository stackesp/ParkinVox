"""
quality.py

EN: This module contains functions for quality assessment of audio.
TR: Bu modül ses kalitesi değerlendirmesi için fonksiyonlar içeriyor.
"""

from typing import Any

import librosa
import numpy as np

from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps


def calculate_quality_features(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> dict[str, Any]:
    """
    EN: Calculate quality features before Parkinson feature extraction.
    TR: Parkinson özelliklerini çıkarmadan önce kalite özelliklerini hesapla.
    """

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("[QUALITY_ERR] Audio must be a non-empty mono array.")

    squared_samples = np.square(audio)
    mean_squared_amplitude = np.mean(squared_samples)
    rms_amplitude = np.sqrt(mean_squared_amplitude)

    absolute_samples = np.abs(audio)
    peak_amplitude = np.max(absolute_samples)

    average_amplitude = np.mean(audio)
    centered_audio = audio - average_amplitude

    clipping_mask = absolute_samples >= config.clipping_level
    clipping_fraction = np.mean(clipping_mask)

    frame_length = min(config.frame_length, len(centered_audio))
    hop_length = min(config.hop_length, frame_length)

    frame_rms_values = librosa.feature.rms(y=centered_audio, frame_length=frame_length, hop_length=hop_length)[0]
    reference_amplitude = max(float(np.max(frame_rms_values)), EPSILON)
    relative_frame_decibels = librosa.amplitude_to_db(frame_rms_values + EPSILON, ref=reference_amplitude, top_db=None)
    absolute_frame_decibels = librosa.amplitude_to_db(frame_rms_values + EPSILON, ref=1.0, top_db=None)
    active_frame_mask = (relative_frame_decibels > -config.silence_threshold_db) & (absolute_frame_decibels > config.low_level_threshold_dbfs)
    active_frame_fraction = np.mean(active_frame_mask)
    silence_fraction = 1.0 - active_frame_fraction
    duration_seconds = len(audio) / sample_rate
    estimated_voiced_seconds = active_frame_fraction * duration_seconds
    low_level_threshold = 10 ** (config.low_level_threshold_dbfs / 20)
    low_level_flag = rms_amplitude < low_level_threshold
    clipping_flag = clipping_fraction > config.maximum_clipping_fraction
    short_recording_flag = estimated_voiced_seconds < config.minimum_voiced_seconds
    
    return {
        "qc_peak_amplitude": float(peak_amplitude),
        "qc_rms_amplitude": float(rms_amplitude),
        "qc_dc_offset": float(average_amplitude),
        "qc_clipping_fraction": float(clipping_fraction),
        "qc_silence_fraction": float(silence_fraction),
        "qc_estimated_voiced_seconds": float(estimated_voiced_seconds),
        "qc_low_level_flag": bool(low_level_flag),
        "qc_clipping_flag": bool(clipping_flag),
        "qc_short_recording_flag": bool(short_recording_flag),
    }
