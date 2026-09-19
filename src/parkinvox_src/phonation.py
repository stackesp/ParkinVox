"""
phonation.py
"""

from typing import Any

import librosa
import numpy as np

from parkinvox_src.audio import estimate_f0_values
from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _perturbation_quotient(values: np.ndarray, window_size: int) -> float:
    if len(values) < window_size:
        return 0.0

    local_averages = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    half_window = window_size // 2
    centered_values = values[half_window:len(values) - half_window]
    denominator = _safe_mean(values) + EPSILON
    return float(np.mean(np.abs(centered_values - local_averages)) / denominator)


def _calculate_cpps_approximation(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> float:
    frame_length = min(config.frame_length, len(audio))
    if frame_length < 256:
        return 0.0

    spectrum = np.abs(np.fft.rfft(audio[:frame_length] * np.hanning(frame_length)))
    log_spectrum = np.log(spectrum + EPSILON)
    cepstrum = np.abs(np.fft.irfft(log_spectrum))

    minimum_quefrency = max(1, int(sample_rate / config.maximum_f0_hz))
    maximum_quefrency = min(len(cepstrum) - 1, int(sample_rate / config.minimum_f0_hz))
    if maximum_quefrency <= minimum_quefrency:
        return 0.0

    quefrencies = np.arange(minimum_quefrency, maximum_quefrency + 1)
    selected_cepstrum = cepstrum[minimum_quefrency:maximum_quefrency + 1]
    regression = np.polyfit(quefrencies, selected_cepstrum, 1)
    baseline = np.polyval(regression, quefrencies)
    prominence = np.max(selected_cepstrum - baseline)
    return float(20 * np.log10(prominence + EPSILON))


def calculate_phonation_features(
    audio: np.ndarray,
    sample_rate: int,
    config: ExtractionConfig,
) -> dict[str, Any]:
    """
    EN: Extract F0, jitter, shimmer, HNR, CPPS and intensity features.
    TR: F0, jitter, shimmer, HNR, CPPS ve şiddet özelliklerini çıkar.
    """

    f0_values = estimate_f0_values(audio, sample_rate, config.minimum_f0_hz, config.maximum_f0_hz, config.frame_length, config.hop_length)
    valid_f0_values = f0_values[np.isfinite(f0_values)]
    valid_f0_fraction = len(valid_f0_values) / max(len(f0_values), 1)

    periods = 1.0 / valid_f0_values if len(valid_f0_values) else np.array([])
    local_jitter = (
        _safe_mean(np.abs(np.diff(periods))) / (_safe_mean(periods) + EPSILON)
        if len(periods) > 1 else 0.0
    )

    rms_values = librosa.feature.rms(y=audio, frame_length=config.frame_length, hop_length=config.hop_length)[0]
    voiced_rms_values = rms_values[:len(f0_values)][np.isfinite(f0_values[:len(rms_values)])]
    local_shimmer = (
        _safe_mean(np.abs(np.diff(voiced_rms_values))) /
        (_safe_mean(voiced_rms_values) + EPSILON)
        if len(voiced_rms_values) > 1 else 0.0
    )

    harmonic_audio = librosa.effects.harmonic(audio)
    noise_audio = audio - harmonic_audio
    harmonic_power = np.mean(np.square(harmonic_audio))
    noise_power = np.mean(np.square(noise_audio))
    hnr_db = 10 * np.log10((harmonic_power + EPSILON) / (noise_power + EPSILON))

    intensity_db = librosa.amplitude_to_db(rms_values + EPSILON, ref=1.0)

    return {
        "phon_f0_mean_hz": _safe_mean(valid_f0_values),
        "phon_f0_median_hz": float(np.median(valid_f0_values)) if len(valid_f0_values) else 0.0,
        "phon_f0_std_hz": float(np.std(valid_f0_values)) if len(valid_f0_values) else 0.0,
        "phon_f0_iqr_hz": float(np.subtract(*np.percentile(valid_f0_values, [75, 25]))) if len(valid_f0_values) else 0.0,
        "phon_valid_f0_fraction": float(valid_f0_fraction),
        "phon_jitter_local": float(local_jitter),
        "phon_jitter_rap": _perturbation_quotient(periods, 3),
        "phon_jitter_ppq5": _perturbation_quotient(periods, 5),
        "phon_shimmer_local": float(local_shimmer),
        "phon_shimmer_apq3": _perturbation_quotient(voiced_rms_values, 3),
        "phon_shimmer_apq5": _perturbation_quotient(voiced_rms_values, 5),
        "phon_shimmer_apq11": _perturbation_quotient(voiced_rms_values, 11),
        "phon_hnr_db": float(hnr_db),
        "phon_cpps_approx_db": _calculate_cpps_approximation(audio, sample_rate, config),
        "phon_intensity_mean_dbfs": _safe_mean(intensity_db),
        "phon_intensity_std_db": float(np.std(intensity_db)),
        "phon_maximum_phonation_time_seconds": float(len(audio) / sample_rate),
    }
