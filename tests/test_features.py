from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from parkinvox_src.audio import load_audio, remove_dc_offset
from parkinvox_src.config import ExtractionConfig
from parkinvox_src.features import extract_voice_features
from parkinvox_src.quality import calculate_quality_features


CONFIG = ExtractionConfig()
FLOAT_EPSILON = 1e-12


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16_000) -> Path:
    """Write floating-point samples without introducing PCM quantization."""
    sf.write(path, np.asarray(audio, dtype=np.float64), sample_rate, subtype="FLOAT")
    return path


def sine_wave(
    duration_seconds: float = 2.0,
    sample_rate: int = 16_000,
    frequency_hz: float = 180.0,
    amplitude: float = 0.2,
    dc_offset: float = 0.0,
) -> np.ndarray:
    times = np.arange(round(duration_seconds * sample_rate), dtype=np.float64) / sample_rate
    return dc_offset + amplitude * np.sin(2.0 * np.pi * frequency_hz * times)


def ddk_like_signal(
    duration_seconds: float = 3.0,
    sample_rate: int = 16_000,
    repetitions_per_second: float = 4.0,
) -> np.ndarray:
    """Create separated voiced bursts with known repetition timing."""
    times = np.arange(round(duration_seconds * sample_rate), dtype=np.float64) / sample_rate
    carrier = np.sin(2.0 * np.pi * 180.0 * times)
    envelope = np.zeros_like(times)

    # Start away from t=0 so every burst has a clear onset.
    for center in np.arange(0.20, duration_seconds - 0.05, 1.0 / repetitions_per_second):
        # A 70 ms Gaussian burst is sufficiently separated at 4 repetitions/s.
        envelope += np.exp(-0.5 * ((times - center) / 0.025) ** 2)

    envelope = np.clip(envelope, 0.0, 1.0)
    return 0.35 * carrier * envelope


def assert_all_numeric_values_finite(features: dict[str, Any]) -> None:
    for name, value in features.items():
        if isinstance(value, (float, np.floating)):
            assert math.isfinite(float(value)), f"{name} is not finite: {value!r}"
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            assert math.isfinite(float(value)), f"{name} is not finite: {value!r}"


def assert_common_invariants(features: dict[str, Any]) -> None:
    assert features["source_sample_rate_hz"] > 0
    assert features["source_channels"] >= 1
    assert features["source_frames"] > 0
    assert features["duration_seconds"] > 0
    assert features["analysis_sample_rate_hz"] == CONFIG.analysis_sample_rate
    assert features["has_invalid_values"] is False

    assert 0.0 <= features["qc_peak_amplitude"]
    assert 0.0 <= features["qc_rms_amplitude"]
    assert 0.0 <= features["qc_clipping_fraction"] <= 1.0
    assert 0.0 <= features["qc_silence_fraction"] <= 1.0
    assert 0.0 <= features["qc_estimated_voiced_seconds"] <= features["duration_seconds"]
    assert_all_numeric_values_finite(features)


# ---------------------------------------------------------------------------
# Audio loading and preprocessing
# ---------------------------------------------------------------------------


def test_load_audio_reports_exact_metadata(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "known.wav", sine_wave(duration_seconds=1.25))
    audio, sample_rate, metadata = load_audio(path)

    assert sample_rate == 16_000
    assert len(audio) == 20_000
    assert metadata == {
        "source_sample_rate_hz": 16_000,
        "source_channels": 1,
        "source_frames": 20_000,
        "duration_seconds": pytest.approx(1.25),
        "has_invalid_values": False,
    }


def test_stereo_metadata_and_first_channel_selection(tmp_path: Path) -> None:
    left = sine_wave(frequency_hz=180.0)
    right = sine_wave(frequency_hz=300.0)
    path = write_wav(tmp_path / "stereo.wav", np.column_stack([left, right]))

    audio, sample_rate, metadata = load_audio(path)

    assert sample_rate == 16_000
    assert metadata["source_channels"] == 2
    assert np.allclose(audio, left, atol=1e-7)


def test_empty_audio_is_rejected(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "empty.wav", np.array([], dtype=np.float64))
    with pytest.raises(ValueError, match="empty"):
        load_audio(path)


def test_remove_dc_offset_centers_signal() -> None:
    audio = sine_wave(dc_offset=0.125)
    centered = remove_dc_offset(audio)

    assert float(np.mean(centered)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(centered)) == pytest.approx(float(np.std(audio)), rel=1e-12)


# ---------------------------------------------------------------------------
# Quality-control ground truth
# ---------------------------------------------------------------------------


def test_clean_sine_quality_matches_analytic_values() -> None:
    audio = sine_wave(amplitude=0.2)
    quality = calculate_quality_features(audio, 16_000, CONFIG)

    assert quality["qc_peak_amplitude"] == pytest.approx(0.2, abs=1e-10)
    assert quality["qc_rms_amplitude"] == pytest.approx(0.2 / math.sqrt(2.0), rel=1e-6)
    assert quality["qc_dc_offset"] == pytest.approx(0.0, abs=1e-12)
    assert quality["qc_clipping_fraction"] == pytest.approx(0.0)
    assert quality["qc_low_level_flag"] is False
    assert quality["qc_clipping_flag"] is False
    assert quality["qc_short_recording_flag"] is False


def test_silence_is_detected() -> None:
    quality = calculate_quality_features(np.zeros(32_000), 16_000, CONFIG)

    assert quality["qc_peak_amplitude"] == 0.0
    assert quality["qc_rms_amplitude"] == 0.0
    assert quality["qc_silence_fraction"] > 0.99
    assert quality["qc_estimated_voiced_seconds"] == pytest.approx(0.0)
    assert quality["qc_low_level_flag"] is True
    assert quality["qc_short_recording_flag"] is True


def test_quiet_signal_is_low_level() -> None:
    audio = sine_wave(amplitude=0.001)
    quality = calculate_quality_features(audio, 16_000, CONFIG)
    assert quality["qc_low_level_flag"] is True


def test_clipping_fraction_and_flag() -> None:
    audio = sine_wave(amplitude=1.2)
    audio = np.clip(audio, -1.0, 1.0)
    quality = calculate_quality_features(audio, 16_000, CONFIG)

    expected_fraction = float(np.mean(np.abs(audio) >= CONFIG.clipping_level))
    assert quality["qc_clipping_fraction"] == pytest.approx(expected_fraction)
    assert expected_fraction > CONFIG.maximum_clipping_fraction
    assert quality["qc_clipping_flag"] is True


def test_short_recording_is_flagged() -> None:
    audio = sine_wave(duration_seconds=0.20)
    quality = calculate_quality_features(audio, 16_000, CONFIG)
    assert quality["qc_short_recording_flag"] is True


# ---------------------------------------------------------------------------
# End-to-end sustained-vowel validation
# ---------------------------------------------------------------------------


def test_sustained_sine_has_correct_f0_and_schema(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "vowel_like.wav", sine_wave(frequency_hz=180.0))
    features = extract_voice_features(path, task="sustained_vowel")

    assert_common_invariants(features)
    assert features["recording_task"] == "sustained_vowel"
    assert features["phon_f0_median_hz"] == pytest.approx(180.0, abs=3.0)
    assert features["phon_f0_mean_hz"] == pytest.approx(180.0, abs=3.0)
    assert features["phon_f0_std_hz"] < 3.0
    assert features["phon_valid_f0_fraction"] > 0.95
    assert features["phon_jitter_local"] < 0.03
    assert features["phon_shimmer_local"] < 0.03
    assert features["phon_maximum_phonation_time_seconds"] == pytest.approx(2.0)

    assert "art_mfcc_01_mean" in features
    assert "art_mfcc_delta_13_std" in features
    assert "art_mfcc_delta2_13_std" in features
    assert "nonlinear_dfa" in features
    assert "pros_pause_ratio" not in features
    assert "time_syllable_nuclei_count" not in features


def test_f0_is_stable_across_source_sample_rates(tmp_path: Path) -> None:
    path_8k = write_wav(
        tmp_path / "tone_8k.wav",
        sine_wave(sample_rate=8_000, frequency_hz=180.0),
        sample_rate=8_000,
    )
    path_16k = write_wav(
        tmp_path / "tone_16k.wav",
        sine_wave(sample_rate=16_000, frequency_hz=180.0),
        sample_rate=16_000,
    )

    features_8k = extract_voice_features(path_8k, task="sustained_vowel")
    features_16k = extract_voice_features(path_16k, task="sustained_vowel")

    assert features_8k["source_sample_rate_hz"] == 8_000
    assert features_8k["analysis_sample_rate_hz"] == 16_000
    assert features_8k["phon_f0_median_hz"] == pytest.approx(180.0, abs=3.0)
    assert features_8k["phon_f0_median_hz"] == pytest.approx(
        features_16k["phon_f0_median_hz"], abs=2.0
    )


def test_amplitude_scaling_preserves_f0_and_changes_dbfs(tmp_path: Path) -> None:
    quiet_path = write_wav(tmp_path / "a01.wav", sine_wave(amplitude=0.1))
    loud_path = write_wav(tmp_path / "a02.wav", sine_wave(amplitude=0.2))

    quiet = extract_voice_features(quiet_path, task="sustained_vowel")
    loud = extract_voice_features(loud_path, task="sustained_vowel")

    assert quiet["phon_f0_median_hz"] == pytest.approx(loud["phon_f0_median_hz"], abs=0.5)
    assert quiet["art_spectral_centroid_mean_hz"] == pytest.approx(
        loud["art_spectral_centroid_mean_hz"], rel=0.02
    )

    expected_gain_db = 20.0 * math.log10(2.0)
    measured_gain_db = loud["phon_intensity_mean_dbfs"] - quiet["phon_intensity_mean_dbfs"]
    assert measured_gain_db == pytest.approx(expected_gain_db, abs=0.15)


def test_dc_offset_does_not_change_prepared_pitch(tmp_path: Path) -> None:
    centered_path = write_wav(tmp_path / "centered.wav", sine_wave())
    offset_path = write_wav(tmp_path / "offset.wav", sine_wave(dc_offset=0.1))

    centered = extract_voice_features(centered_path, task="sustained_vowel")
    offset = extract_voice_features(offset_path, task="sustained_vowel")

    assert offset["qc_dc_offset"] == pytest.approx(0.1, abs=1e-7)
    assert centered["phon_f0_median_hz"] == pytest.approx(
        offset["phon_f0_median_hz"], abs=0.5
    )


# ---------------------------------------------------------------------------
# DDK and reading-task validation
# ---------------------------------------------------------------------------


def test_ddk_like_bursts_produce_plausible_onset_rate(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "ddk_like.wav", ddk_like_signal())
    features = extract_voice_features(path, task="ddk")

    assert_common_invariants(features)
    assert features["recording_task"] == "ddk"
    assert "time_syllable_nuclei_count" in features
    assert "pros_pause_ratio" not in features
    assert "phon_f0_median_hz" not in features

    # This is deliberately a tolerance, not exact equality: onset detectors
    # operate on frames and may miss or merge a boundary burst.
    assert 8 <= features["time_syllable_nuclei_count"] <= 13
    assert 2.5 <= features["time_estimated_syllables_per_second"] <= 4.5
    assert features["time_interval_mean_seconds"] == pytest.approx(0.25, abs=0.08)
    assert features["time_interval_cv"] < 0.25


def test_reading_duration_invariants(tmp_path: Path) -> None:
    """This should fail until padded-frame speech duration is corrected."""
    path = write_wav(tmp_path / "reading_like.wav", ddk_like_signal(duration_seconds=3.0))
    features = extract_voice_features(path, task="reading")

    assert_common_invariants(features)
    duration = features["pros_duration_seconds"]

    assert features["recording_task"] == "reading"
    assert 0.0 <= features["pros_speech_seconds"] <= duration
    assert 0.0 <= features["pros_pause_total_seconds"] <= duration
    assert 0.0 <= features["pros_pause_ratio"] <= 1.0
    assert 0.0 <= features["pros_voiced_frame_fraction"] <= 1.0
    assert features["pros_pause_total_seconds"] + features["pros_speech_seconds"] <= (
        duration + CONFIG.hop_length / CONFIG.analysis_sample_rate
    )


def test_reading_and_spontaneous_have_expected_schema(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "speech_like.wav", ddk_like_signal())

    for task in ("reading", "spontaneous"):
        features = extract_voice_features(path, task=task)
        assert_common_invariants(features)
        assert features["recording_task"] == task
        assert "pros_pause_ratio" in features
        assert "time_syllable_nuclei_count" in features
        assert "art_mfcc_01_mean" in features
        assert "phon_jitter_local" not in features
        assert "nonlinear_dfa" not in features


# ---------------------------------------------------------------------------
# Robustness, errors, and repeatability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["sustained_vowel", "ddk", "reading", "spontaneous"])
def test_all_tasks_return_only_finite_numeric_values(tmp_path: Path, task: str) -> None:
    path = write_wav(tmp_path / f"{task}.wav", ddk_like_signal())
    features = extract_voice_features(path, task=task)  # type: ignore[arg-type]
    assert_common_invariants(features)


def test_unsupported_task_is_rejected(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "valid.wav", sine_wave())
    with pytest.raises(ValueError, match="Unsupported recording task"):
        extract_voice_features(path, task="not_a_task")  # type: ignore[arg-type]


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, RuntimeError)):
        extract_voice_features(tmp_path / "missing.wav", task="sustained_vowel")


def test_extraction_is_deterministic(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "repeatable.wav", sine_wave())

    first = extract_voice_features(path, task="sustained_vowel")
    second = extract_voice_features(path, task="sustained_vowel")

    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], float):
            assert first[key] == pytest.approx(second[key], rel=0.0, abs=FLOAT_EPSILON)
        else:
            assert first[key] == second[key]


def test_feature_names_are_unique_and_well_formed(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "schema.wav", sine_wave())
    features = extract_voice_features(path, task="sustained_vowel")

    assert len(features) == len(set(features))
    for name in features:
        assert name == name.strip()
        assert name == name.lower()
        assert " " not in name


def test_sustained_vowel_feature_count_is_stable(tmp_path: Path) -> None:
    """Catches accidental feature deletion/addition; update intentionally."""
    path = write_wav(tmp_path / "schema_count.wav", sine_wave())
    features = extract_voice_features(path, task="sustained_vowel")

    # Current schema: 5 metadata + 9 QC + 2 task/analysis + 17 phonation
    # + 85 articulation + 5 nonlinear = 123 total fields.
    assert len(features) == 123
