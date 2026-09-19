"""
timing.py

EN: Timing and rapid syllable repetition features.
TR: Zamanlama ve hızlı hece tekrarı özellikleri.

I think this might not be the best way to do this tbh.
"""

from typing import Any

import librosa
import numpy as np

from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps


def calculate_timing_features(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> dict[str, Any]:
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=config.hop_length)
    threshold = np.median(onset_envelope) + np.std(onset_envelope)
    candidate_frames = np.flatnonzero(
        (onset_envelope[1:-1] > onset_envelope[:-2])
        & (onset_envelope[1:-1] >= onset_envelope[2:])
        & (onset_envelope[1:-1] > threshold)
    ) + 1
    onset_frames = candidate_frames
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=config.hop_length)

    duration_seconds = len(audio) / sample_rate
    intervals = np.diff(onset_times)
    interval_mean = float(np.mean(intervals)) if len(intervals) else 0.0
    interval_std = float(np.std(intervals)) if len(intervals) else 0.0
    interval_cv = interval_std / (interval_mean + EPSILON)

    if len(intervals) >= 2:
        interval_numbers = np.arange(len(intervals))
        tempo_slope = float(np.polyfit(interval_numbers, intervals, 1)[0])
    else:
        tempo_slope = 0.0

    return {
        "time_syllable_nuclei_count": int(len(onset_times)),
        "time_estimated_syllables_per_second": float(len(onset_times) / max(duration_seconds, EPSILON)),
        "time_interval_mean_seconds": interval_mean,
        "time_interval_std_seconds": interval_std,
        "time_interval_cv": float(interval_cv),
        "time_interval_slope_seconds": tempo_slope,
        "time_first_onset_seconds": float(onset_times[0]) if len(onset_times) else duration_seconds,
    }
