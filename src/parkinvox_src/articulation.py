"""
articulation.py

EN: Articulation, spectral, MFCC and vowel features.
TR: Artikülasyon, spektral, MFCC ve ünlü özellikleri.
"""

from typing import Any

import librosa
import numpy as np

from parkinvox_src.config import ExtractionConfig

EPSILON = np.finfo(float).eps

def _summarize_matrix(prefix: str, values: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for index in range(values.shape[0]):
        features[f"{prefix}_{index + 1:02d}_mean"] = float(np.mean(values[index]))
        features[f"{prefix}_{index + 1:02d}_std"] = float(np.std(values[index]))
    return features

def _estimate_formants(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> tuple[float, float, float]:
    emphasized_audio = librosa.effects.preemphasis(audio)
    frame_length = min(config.frame_length, len(emphasized_audio))
    if frame_length <= config.lpc_order + 2:
        return 0.0, 0.0, 0.0

    frames = librosa.util.frame(emphasized_audio, frame_length=frame_length, hop_length=min(config.hop_length, frame_length))
    formant_rows: list[list[float]] = []

    for frame in frames.T:
        # Each frame is a 1D array
        if np.sqrt(np.mean(np.square(frame))) < 1e-4:
            continue

        try:
            coefficients = librosa.lpc(frame * np.hamming(len(frame)), order=config.lpc_order)
            roots = np.roots(coefficients)
            roots = roots[np.imag(roots) >= 0]
            frequencies = np.sort(np.angle(roots) * sample_rate / (2 * np.pi))
            frequencies = frequencies[(frequencies > 90) & (frequencies < 5000)]
            if len(frequencies) >= 3:
                formant_rows.append(frequencies[:3].tolist())  
        except (FloatingPointError, np.linalg.LinAlgError):
            continue

    if not formant_rows:
        return 0.0, 0.0, 0.0

    medians = np.median(np.asarray(formant_rows), axis=0)
    return float(medians[0]), float(medians[1]), float(medians[2])

def calculate_articulation_features(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> dict[str, Any]:
    """
    EN: Extract MFCC, delta, spectrum and formant features.
    TR: MFCC, delta, spektrum ve formant özelliklerini çıkarır.
    """

    mfcc_values = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=config.number_of_mfccs, n_fft=config.frame_length, hop_length=config.hop_length)
    mfcc_delta_values = librosa.feature.delta(mfcc_values, mode="nearest")
    mfcc_delta2_values = librosa.feature.delta(mfcc_values, order=2, mode="nearest")

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]

    first_formant, second_formant, third_formant = _estimate_formants(audio, sample_rate, config)

    features: dict[str, Any] = {
        "art_formant_f1_median_hz": first_formant,
        "art_formant_f2_median_hz": second_formant,
        "art_formant_f3_median_hz": third_formant,
        "art_spectral_centroid_mean_hz": float(np.mean(spectral_centroid)),
        "art_spectral_bandwidth_mean_hz": float(np.mean(spectral_bandwidth)),
        "art_spectral_rolloff_mean_hz": float(np.mean(spectral_rolloff)),
        "art_spectral_flatness_mean": float(np.mean(spectral_flatness)),
    }
    features.update(_summarize_matrix("art_mfcc", mfcc_values))
    features.update(_summarize_matrix("art_mfcc_delta", mfcc_delta_values))
    features.update(_summarize_matrix("art_mfcc_delta2", mfcc_delta2_values))

    return features

def calculate_vowel_features(vowel_formants: dict[str, tuple[float, float]]) -> dict[str, float]:
    """
    EN: Calculate vowel features from the formants of the vowels.
    TR: Ünlülerin formantlarından ünlü özelliklerini hesaplar.
    """

    required_vowels = {"a", "i", "u"}
    if not required_vowels.issubset(vowel_formants):
        return {"art_vsa_hz2": 0.0, "art_vai": 0.0, "art_fcr": 0.0}

    f1_a, f2_a = vowel_formants["a"]
    f1_i, f2_i = vowel_formants["i"]
    f1_u, f2_u = vowel_formants["u"]

    # VSA (Vowel Space Area) calculation
    # VSA hesaplaması
    vowel_space_area = 0.5 * abs(
        f1_a * (f2_i - f2_u) +
        f1_i * (f2_u - f2_a) +
        f1_u * (f2_a - f2_i)
    )

    vowel_arti_index = (f2_i + f1_a) / (f1_i + f1_u + f2_u + f2_a + EPSILON)
    formant_cent_ratio = 1.0 / (vowel_arti_index + EPSILON)

    vowel_space_area = float(vowel_space_area)
    vowel_arti_index = float(vowel_arti_index)
    formant_cent_ratio = float(formant_cent_ratio)

    return {
        "art_vsa_hz2": vowel_space_area,
        "art_vai": vowel_arti_index,
        "art_fcr": formant_cent_ratio,
    }