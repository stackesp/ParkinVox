"""
features.py

EN: Main interface for extracting Parkinson voice feature groups.
TR: Parkinson ses özellik gruplarını çıkarmak için ana arayüz.
"""

from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np

from parkinvox_src.articulation import calculate_articulation_features
from parkinvox_src.audio import load_audio, remove_dc_offset
from parkinvox_src.config import ExtractionConfig
from parkinvox_src.nonlinear import calculate_nonlinear_features
from parkinvox_src.phonation import calculate_phonation_features
from parkinvox_src.prosody import calculate_prosody_features
from parkinvox_src.quality import calculate_quality_features
from parkinvox_src.timing import calculate_timing_features

RecordingTask = Literal["sustained_vowel", "ddk", "reading", "spontaneous"]


def prepare_audio(audio: np.ndarray, sample_rate: int, config: ExtractionConfig) -> tuple[np.ndarray, int]:
    """
    EN: Remove DC offset and resample without loudness normalization.
    TR: Ses yüksekliğini normalize etmeden DC ofsetini kaldır ve yeniden örnekle.
    """

    centered_audio = remove_dc_offset(audio)
    if sample_rate != config.analysis_sample_rate:
        centered_audio = librosa.resample(centered_audio, orig_sr=sample_rate, target_sr=config.analysis_sample_rate)
        sample_rate = config.analysis_sample_rate
    return centered_audio.astype(np.float64), sample_rate


def extract_voice_features(
    audio_path: str | Path,
    task: RecordingTask,
    config: ExtractionConfig | None = None,
) -> dict[str, Any]:
    """
    EN: Extract quality features and task-appropriate Parkinson voice features.
    TR: Kalite ve göreve uygun ses özelliklerini çıkar.
    """

    selected_config = config or ExtractionConfig()
    source_audio, source_sample_rate, metadata = load_audio(audio_path)
    quality_features = calculate_quality_features(source_audio, source_sample_rate, selected_config)
    audio, sample_rate = prepare_audio(source_audio, source_sample_rate, selected_config)

    features: dict[str, Any] = {
        **metadata,
        **quality_features,
        "recording_task": task,
        "analysis_sample_rate_hz": sample_rate,
    }

    if task == "sustained_vowel":
        features.update(calculate_phonation_features(audio, sample_rate, selected_config))
        features.update(calculate_articulation_features(audio, sample_rate, selected_config))
        features.update(calculate_nonlinear_features(audio, sample_rate, selected_config))
    elif task == "ddk":
        features.update(calculate_timing_features(audio, sample_rate, selected_config))
        features.update(calculate_articulation_features(audio, sample_rate, selected_config))
    elif task in {"reading", "spontaneous"}:
        features.update(calculate_prosody_features(audio, sample_rate, selected_config))
        features.update(calculate_timing_features(audio, sample_rate, selected_config))
        features.update(calculate_articulation_features(audio, sample_rate, selected_config))
    else:
        raise ValueError(f"[FEATURE_ERR] Unsupported recording task: {task}")

    return features
