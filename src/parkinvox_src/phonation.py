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


def _refine_peak(scores: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(scores) - 1:
        return float(index)

    left, centre, right = scores[index - 1], scores[index], scores[index + 1]
    denominator = left - 2 * centre + right
    if abs(denominator) < EPSILON:
        return float(index)

    return float(index + 0.5 * (left - right) / denominator)


def _trace_cycles(
    audio: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    expected_period_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    segment_start = max(0, start_sample)
    segment_end = min(len(audio), end_sample)
    expected_lag = expected_period_seconds * sample_rate
    previous_lag = expected_lag
    if segment_end - segment_start < 5 * previous_lag:
        return np.array([]), np.array([])

    active = np.flatnonzero(np.abs(audio[segment_start:segment_end]) > 0.1 * np.max(np.abs(audio[segment_start:segment_end])))
    if len(active) == 0:
        return np.array([]), np.array([])

    seed_start = segment_start + int(active[0] + previous_lag)
    seed_region = audio[seed_start:seed_start + int(previous_lag)]
    position = float(seed_start + np.argmax(seed_region))
    periods: list[float] = []
    amplitudes: list[float] = []

    while True:
        half_window = int(round(previous_lag / 2))
        centre = int(round(position))
        minimum_lag = int(round(previous_lag * 0.75))
        maximum_lag = int(round(previous_lag * 1.25))
        window_start = centre - half_window
        window_end = centre + maximum_lag + half_window + 2
        if window_start < segment_start or window_end > segment_end:
            break

        reference = audio[window_start:centre + half_window]
        candidates = np.lib.stride_tricks.sliding_window_view(audio[window_start:window_end], len(reference))
        lags = np.arange(minimum_lag - 1, maximum_lag + 2)
        selected = candidates[lags]
        selected = selected - selected.mean(axis=1, keepdims=True)
        centred_reference = reference - reference.mean()
        scores = selected @ centred_reference / (np.linalg.norm(selected, axis=1) * np.linalg.norm(centred_reference) + EPSILON)

        best_index = int(np.argmax(scores[1:-1])) + 1
        refined_lag = float(lags[0]) + _refine_peak(scores, best_index)
        if not 0.5 * expected_lag <= refined_lag <= 1.5 * expected_lag:
            break

        periods.append(refined_lag / sample_rate)
        amplitudes.append(float(np.ptp(reference)))
        previous_lag = refined_lag
        position += refined_lag

    return np.asarray(periods), np.asarray(amplitudes)


def _voiced_runs(f0_values: np.ndarray) -> list[tuple[int, int]]:
    voiced = np.isfinite(f0_values)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, is_voiced in enumerate(voiced):
        if is_voiced and run_start is None:
            run_start = index
        elif not is_voiced and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(voiced) - 1))

    return runs


def _extract_cycle_series(
    audio: np.ndarray,
    sample_rate: int,
    f0_values: np.ndarray,
    config: ExtractionConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    period_runs: list[np.ndarray] = []
    amplitude_runs: list[np.ndarray] = []
    minimum_period = 1.0 / config.maximum_f0_hz
    maximum_period = 1.0 / config.minimum_f0_hz

    for first_frame, last_frame in _voiced_runs(f0_values):
        expected_period = 1.0 / float(np.median(f0_values[first_frame:last_frame + 1]))
        start_sample = first_frame * config.hop_length + config.frame_length // 2 - config.hop_length
        end_sample = last_frame * config.hop_length + config.frame_length // 2 + config.hop_length
        periods, amplitudes = _trace_cycles(audio, sample_rate, start_sample, end_sample, expected_period)
        valid = (periods >= minimum_period) & (periods <= maximum_period)
        if np.count_nonzero(valid) < 3:
            continue

        period_runs.append(periods[valid])
        amplitude_runs.append(amplitudes[valid])

    return period_runs, amplitude_runs


def _local_perturbation(runs: list[np.ndarray], maximum_factor: float) -> float:
    difference_sum = 0.0
    pair_count = 0
    value_sum = 0.0
    value_count = 0

    for values in runs:
        ratios = np.maximum(values[1:], values[:-1]) / (np.minimum(values[1:], values[:-1]) + EPSILON)
        valid_pairs = ratios <= maximum_factor
        difference_sum += float(np.sum(np.abs(np.diff(values))[valid_pairs]))
        pair_count += int(np.count_nonzero(valid_pairs))
        value_sum += float(np.sum(values))
        value_count += len(values)

    if pair_count == 0 or value_count == 0:
        return 0.0

    return (difference_sum / pair_count) / (value_sum / value_count + EPSILON)


def _run_quotient(runs: list[np.ndarray], window_size: int) -> float:
    usable_runs = [values for values in runs if len(values) >= window_size]
    if not usable_runs:
        return 0.0

    weights = [len(values) for values in usable_runs]
    quotients = [_perturbation_quotient(values, window_size) for values in usable_runs]
    return float(np.average(quotients, weights=weights))


def _calculate_hnr_db(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> float:
    frame_length = config.hnr_frame_length
    window = np.hanning(frame_length)
    window_autocorrelation = np.correlate(window, window, mode="full")[frame_length - 1:]
    window_autocorrelation = window_autocorrelation / window_autocorrelation[0]
    minimum_lag = max(1, int(sample_rate / config.maximum_f0_hz))
    maximum_lag = min(frame_length // 2, int(sample_rate / config.minimum_f0_hz))
    if maximum_lag <= minimum_lag:
        return 0.0

    hnr_values: list[float] = []
    for start in range(0, len(audio) - frame_length + 1, config.hop_length):
        frame = audio[start:start + frame_length]
        frame = (frame - np.mean(frame)) * window
        if np.dot(frame, frame) <= EPSILON:
            continue

        autocorrelation = np.correlate(frame, frame, mode="full")[frame_length - 1:]
        autocorrelation = autocorrelation / autocorrelation[0]
        corrected = autocorrelation[minimum_lag:maximum_lag + 1] / window_autocorrelation[minimum_lag:maximum_lag + 1]
        periodicity = min(float(np.max(corrected)), 0.9999)
        if periodicity >= config.hnr_minimum_periodicity:
            hnr_values.append(10 * np.log10(periodicity / (1 - periodicity)))

    return _safe_mean(np.asarray(hnr_values))


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

    period_runs, amplitude_runs = _extract_cycle_series(audio, sample_rate, f0_values, config)
    local_jitter = _local_perturbation(period_runs, config.maximum_period_factor)
    local_shimmer = _local_perturbation(amplitude_runs, config.maximum_amplitude_factor)

    rms_values = librosa.feature.rms(y=audio, frame_length=config.frame_length, hop_length=config.hop_length)[0]

    hnr_db = _calculate_hnr_db(audio, sample_rate, config)

    intensity_db = librosa.amplitude_to_db(rms_values + EPSILON, ref=1.0)

    return {
        "phon_f0_mean_hz": _safe_mean(valid_f0_values),
        "phon_f0_median_hz": float(np.median(valid_f0_values)) if len(valid_f0_values) else 0.0,
        "phon_f0_std_hz": float(np.std(valid_f0_values)) if len(valid_f0_values) else 0.0,
        "phon_f0_iqr_hz": float(np.subtract(*np.percentile(valid_f0_values, [75, 25]))) if len(valid_f0_values) else 0.0,
        "phon_valid_f0_fraction": float(valid_f0_fraction),
        "phon_jitter_local": float(local_jitter),
        "phon_jitter_rap": _run_quotient(period_runs, 3),
        "phon_jitter_ppq5": _run_quotient(period_runs, 5),
        "phon_shimmer_local": float(local_shimmer),
        "phon_shimmer_apq3": _run_quotient(amplitude_runs, 3),
        "phon_shimmer_apq5": _run_quotient(amplitude_runs, 5),
        "phon_shimmer_apq11": _run_quotient(amplitude_runs, 11),
        "phon_hnr_db": float(hnr_db),
        "phon_cpps_approx_db": _calculate_cpps_approximation(audio, sample_rate, config),
        "phon_intensity_mean_dbfs": _safe_mean(intensity_db),
        "phon_intensity_std_db": float(np.std(intensity_db)),
        "phon_maximum_phonation_time_seconds": float(len(audio) / sample_rate),
    }
