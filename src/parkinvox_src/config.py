"""
config.py

EN: This module contains configuration for all feature extraction files.
TR: Bu modül tüm özellik çıkarma dosyaları için yapılandırma içeriyor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionConfig:
    """
    EN: Shared settings for quality control and Parkinson voice features.
    TR: Kalite kontrolü ve Parkinson ses özellikleri için ortak ayarlar.
    """

    analysis_sample_rate: int = 16000
    nonlinear_sample_rate: int = 200

    minimum_f0_hz: float = 70.0
    maximum_f0_hz: float = 500.0
    maximum_period_factor: float = 1.3
    maximum_amplitude_factor: float = 1.6

    frame_length: int = 2048
    hop_length: int = 256
    hnr_frame_length: int = 640
    hnr_minimum_periodicity: float = 0.5

    silence_threshold_db: float = 45.0
    minimum_voiced_seconds: float = 0.75
    minimum_valid_f0_fraction: float = 0.20

    clipping_level: float = 0.999
    maximum_clipping_fraction: float = 0.01
    low_level_threshold_dbfs: float = -45.0

    number_of_mfccs: int = 13
    formant_ceiling_hz: float = 5000.0
    number_of_formants: int = 5
    formant_window_seconds: float = 0.025
    formant_hop_seconds: float = 0.01
    formant_pre_emphasis_hz: float = 50.0

    embedding_dimension: int = 5
    embedding_delay: int = 2
    recurrence_rate: float = 0.03
    theiler_window: int = 10
    maximum_nonlinear_points: int = 1500