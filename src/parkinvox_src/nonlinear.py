"""
nonlinear.py

EN: Nonlinear features.
TR: Doğrusal olmayan özellikler.
"""

from math import factorial
from typing import Any

import librosa
import numpy as np

from parkinvox_src.audio import estimate_f0_values
from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps


def _detrended_fluctuation_analysis(values: np.ndarray) -> float:
    if len(values) < 64 or np.std(values) < EPSILON:
        return 0.0

    integrated_values = np.cumsum(values - np.mean(values))
    maximum_scale = max(8, len(values) // 4)
    scales = np.unique(np.logspace(np.log10(8), np.log10(maximum_scale), 10).astype(int))
    valid_scales: list[int] = []
    fluctuations: list[float] = []

    for scale in scales:
        number_of_segments = len(integrated_values) // scale
        if number_of_segments < 2:
            continue
        segment_fluctuations = []
        x_values = np.arange(scale)
        for segment_index in range(number_of_segments):
            segment = integrated_values[segment_index * scale:(segment_index + 1) * scale]
            trend = np.polyval(np.polyfit(x_values, segment, 1), x_values)
            segment_fluctuations.append(np.sqrt(np.mean(np.square(segment - trend))))
        valid_scales.append(scale)
        fluctuations.append(float(np.mean(segment_fluctuations)))

    if len(fluctuations) < 2:
        return 0.0
    return float(np.polyfit(np.log(valid_scales), np.log(np.asarray(fluctuations) + EPSILON), 1)[0])


def _permutation_entropy(values: np.ndarray, order: int = 3, delay: int = 1) -> float:
    # Basically the permutation entropy function, nothing fancy.
    if len(values) < order * delay:
        return 0.0

    patterns: dict[tuple[int, ...], int] = {}
    final_index = len(values) - delay * (order - 1)
    for index in range(final_index):
        window = values[index:index + delay * order:delay]
        pattern = tuple(np.argsort(window, kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    probabilities = np.asarray(list(patterns.values()), dtype=float)
    probabilities /= np.sum(probabilities)
    entropy = -np.sum(probabilities * np.log(probabilities + EPSILON))
    return float(entropy / np.log(factorial(order)))


def _pitch_period_entropy(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> float:
    f0_values = estimate_f0_values(audio, sample_rate, config.minimum_f0_hz, config.maximum_f0_hz, config.frame_length, config.hop_length)
    valid_f0_values = f0_values[np.isfinite(f0_values)]
    if len(valid_f0_values) < 5:
        return 0.0

    semitone_values = 12 * np.log2(valid_f0_values / np.median(valid_f0_values))
    histogram, _ = np.histogram(semitone_values, bins=20, density=False)
    probabilities = histogram[histogram > 0].astype(float)
    probabilities /= np.sum(probabilities)
    entropy = -np.sum(probabilities * np.log(probabilities + EPSILON))
    return float(entropy / np.log(20))


def _recurrence_features(values: np.ndarray, config: ExtractionConfig) -> tuple[float, float]:
    dimension = config.embedding_dimension
    delay = config.embedding_delay
    available_points = len(values) - (dimension - 1) * delay
    if available_points < 20:
        return 0.0, 0.0

    embedded = np.column_stack([values[offset * delay:offset * delay + available_points] for offset in range(dimension)])
    if len(embedded) > config.maximum_nonlinear_points:
        indexes = np.linspace(0, len(embedded) - 1, config.maximum_nonlinear_points).astype(int)
        embedded = embedded[indexes]

    distances = np.sqrt(np.sum(np.square(embedded[:, None, :] - embedded[None, :, :]), axis=2))
    upper_distances = distances[np.triu_indices(len(embedded), k=config.theiler_window + 1)]
    if len(upper_distances) == 0:
        return 0.0, 0.0

    threshold = np.quantile(upper_distances, config.recurrence_rate)
    recurrence_matrix = distances <= threshold
    recurrence_rate_value = float(np.mean(recurrence_matrix))

    diagonal_line_points = 0
    recurrent_points = int(np.sum(recurrence_matrix))
    for diagonal_offset in range(-len(embedded) + 1, len(embedded)):
        diagonal = np.diagonal(recurrence_matrix, offset=diagonal_offset)
        run_lengths = np.diff(np.flatnonzero(np.r_[True, diagonal[:-1] != diagonal[1:], True]))
        run_values = diagonal[np.r_[0, np.cumsum(run_lengths)[:-1]]]
        diagonal_line_points += int(np.sum(run_lengths[(run_values == 1) & (run_lengths >= 2)]))

    determinism = diagonal_line_points / max(recurrent_points, 1)
    return recurrence_rate_value, float(determinism)


def calculate_nonlinear_features(
    audio: np.ndarray,
    sample_rate: int,
    config: ExtractionConfig,
) -> dict[str, Any]:
    """
    EN: Calculate DFA, entropy, PPE and recurrence features.
    TR: DFA, entropi, PPE ve yineleme özelliklerini hesapla.
    """

    nonlinear_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config.nonlinear_sample_rate)
    nonlinear_audio = nonlinear_audio / (np.std(nonlinear_audio) + EPSILON)

    recurrence_rate_value, recurrence_determinism = _recurrence_features(nonlinear_audio,config)

    return {
        "nonlinear_dfa": _detrended_fluctuation_analysis(nonlinear_audio),
        "nonlinear_permutation_entropy": _permutation_entropy(nonlinear_audio),
        "nonlinear_ppe": _pitch_period_entropy(audio, sample_rate, config),
        "nonlinear_recurrence_rate": recurrence_rate_value,
        "nonlinear_recurrence_determinism": recurrence_determinism,
    }
