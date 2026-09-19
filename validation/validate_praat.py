from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from parkinvox_src.config import ExtractionConfig
from parkinvox_src.features import extract_voice_features


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff"}
EPSILON = np.finfo(float).eps


@dataclass(frozen=True)
class AcceptanceTargets:
    f0_mae_hz: float = 5.0
    jitter_median_relative_error: float = 0.15
    shimmer_median_relative_error: float = 0.15
    hnr_mae_db: float = 2.0
    f1_mae_hz: float = 50.0
    f2_mae_hz: float = 100.0
    f3_mae_hz: float = 150.0


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def safe_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else math.nan


def discover_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def praat_pitch_features(
    sound: parselmouth.Sound,
    minimum_f0_hz: float,
    maximum_f0_hz: float,
) -> dict[str, float]:
    pitch = sound.to_pitch_ac(
        time_step=0.0,
        pitch_floor=minimum_f0_hz,
        pitch_ceiling=maximum_f0_hz,
    )
    f0_values = np.asarray(pitch.selected_array["frequency"], dtype=float)
    f0_values = f0_values[np.isfinite(f0_values) & (f0_values > 0.0)]
    return {
        "praat_f0_mean_hz": float(np.mean(f0_values)) if len(f0_values) else math.nan,
        "praat_f0_median_hz": float(np.median(f0_values)) if len(f0_values) else math.nan,
        "praat_f0_std_hz": float(np.std(f0_values)) if len(f0_values) else math.nan,
        "praat_valid_f0_frames": float(len(f0_values)),
    }


def praat_perturbation_features(
    sound: parselmouth.Sound,
    minimum_f0_hz: float,
    maximum_f0_hz: float,
) -> dict[str, float]:
    point_process = call(
        sound,
        "To PointProcess (periodic, cc)",
        minimum_f0_hz,
        maximum_f0_hz,
    )
    jitter_local = call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        minimum_f0_hz,
        maximum_f0_hz,
        1.3,
    )
    shimmer_local = call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        minimum_f0_hz,
        maximum_f0_hz,
        1.3,
        1.6,
    )
    return {
        "praat_jitter_local": finite_float(jitter_local),
        "praat_shimmer_local": finite_float(shimmer_local),
    }


def praat_hnr_feature(
    sound: parselmouth.Sound,
    minimum_f0_hz: float,
) -> dict[str, float]:
    harmonicity = sound.to_harmonicity_cc(
        time_step=0.01,
        minimum_pitch=minimum_f0_hz,
        silence_threshold=0.1,
        periods_per_window=1.0,
    )
    hnr_db = call(harmonicity, "Get mean", 0, 0)
    return {"praat_hnr_db": finite_float(hnr_db)}


def praat_formant_features(sound: parselmouth.Sound) -> dict[str, float]:
    formants = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5.0,
        maximum_formant=5_000.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    start_time = max(float(sound.xmin) + 0.05, float(sound.xmin))
    end_time = min(float(sound.xmax) - 0.05, float(sound.xmax))
    if end_time <= start_time:
        start_time = float(sound.xmin)
        end_time = float(sound.xmax)

    sample_times = np.arange(start_time, end_time, 0.01)
    medians: list[float] = []

    for formant_number in (1, 2, 3):
        values = [
            finite_float(formants.get_value_at_time(formant_number, float(time)))
            for time in sample_times
        ]
        medians.append(safe_median(values))

    return {
        "praat_f1_median_hz": medians[0],
        "praat_f2_median_hz": medians[1],
        "praat_f3_median_hz": medians[2],
    }


def extract_praat_features(
    audio_path: Path,
    config: ExtractionConfig,
) -> dict[str, float]:
    sound = parselmouth.Sound(str(audio_path))
    results: dict[str, float] = {}
    results.update(
        praat_pitch_features(
            sound,
            config.minimum_f0_hz,
            config.maximum_f0_hz,
        )
    )
    results.update(
        praat_perturbation_features(
            sound,
            config.minimum_f0_hz,
            config.maximum_f0_hz,
        )
    )
    results.update(praat_hnr_feature(sound, config.minimum_f0_hz))
    results.update(praat_formant_features(sound))
    return results


def absolute_error(first: float, second: float) -> float:
    if not math.isfinite(first) or not math.isfinite(second):
        return math.nan
    return abs(first - second)


def signed_error(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return math.nan
    return candidate - reference


def relative_error(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        return math.nan
    return abs(candidate - reference) / max(abs(reference), EPSILON)


def compare_recording(
    audio_path: Path,
    config: ExtractionConfig,
) -> dict[str, object]:
    parkinvox = extract_voice_features(
        audio_path,
        task="sustained_vowel",
        config=config,
    )
    praat = extract_praat_features(audio_path, config)

    row: dict[str, object] = {
        "file": str(audio_path),
        "duration_seconds": finite_float(parkinvox["duration_seconds"]),
        "qc_low_level_flag": bool(parkinvox["qc_low_level_flag"]),
        "qc_clipping_flag": bool(parkinvox["qc_clipping_flag"]),
        "qc_short_recording_flag": bool(parkinvox["qc_short_recording_flag"]),
        "parkinvox_f0_median_hz": finite_float(parkinvox["phon_f0_median_hz"]),
        "praat_f0_median_hz": praat["praat_f0_median_hz"],
        "parkinvox_jitter_local": finite_float(parkinvox["phon_jitter_local"]),
        "praat_jitter_local": praat["praat_jitter_local"],
        "parkinvox_shimmer_local": finite_float(parkinvox["phon_shimmer_local"]),
        "praat_shimmer_local": praat["praat_shimmer_local"],
        "parkinvox_hnr_db": finite_float(parkinvox["phon_hnr_db"]),
        "praat_hnr_db": praat["praat_hnr_db"],
        "parkinvox_f1_median_hz": finite_float(parkinvox["art_formant_f1_median_hz"]),
        "praat_f1_median_hz": praat["praat_f1_median_hz"],
        "parkinvox_f2_median_hz": finite_float(parkinvox["art_formant_f2_median_hz"]),
        "praat_f2_median_hz": praat["praat_f2_median_hz"],
        "parkinvox_f3_median_hz": finite_float(parkinvox["art_formant_f3_median_hz"]),
        "praat_f3_median_hz": praat["praat_f3_median_hz"],
    }

    comparisons = {
        "f0": ("parkinvox_f0_median_hz", "praat_f0_median_hz"),
        "jitter": ("parkinvox_jitter_local", "praat_jitter_local"),
        "shimmer": ("parkinvox_shimmer_local", "praat_shimmer_local"),
        "hnr": ("parkinvox_hnr_db", "praat_hnr_db"),
        "f1": ("parkinvox_f1_median_hz", "praat_f1_median_hz"),
        "f2": ("parkinvox_f2_median_hz", "praat_f2_median_hz"),
        "f3": ("parkinvox_f3_median_hz", "praat_f3_median_hz"),
    }

    for name, (candidate_name, reference_name) in comparisons.items():
        candidate = finite_float(row[candidate_name])
        reference = finite_float(row[reference_name])
        row[f"{name}_signed_error"] = signed_error(candidate, reference)
        row[f"{name}_absolute_error"] = absolute_error(candidate, reference)
        row[f"{name}_relative_error"] = relative_error(candidate, reference)

    return row


def finite_series(data_frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(data_frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def paired_values(
    data_frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = pd.to_numeric(data_frame[candidate_column], errors="coerce").to_numpy(dtype=float)
    reference = pd.to_numeric(data_frame[reference_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(candidate) & np.isfinite(reference)
    return candidate[valid], reference[valid]


def correlation(candidate: np.ndarray, reference: np.ndarray) -> float:
    if len(candidate) < 2 or np.std(candidate) <= EPSILON or np.std(reference) <= EPSILON:
        return math.nan
    return float(np.corrcoef(candidate, reference)[0, 1])


def summarize_metric(
    data_frame: pd.DataFrame,
    name: str,
    candidate_column: str,
    reference_column: str,
) -> dict[str, float]:
    candidate, reference = paired_values(data_frame, candidate_column, reference_column)
    differences = candidate - reference
    absolute_errors = np.abs(differences)
    relative_errors = absolute_errors / np.maximum(np.abs(reference), EPSILON)

    if not len(differences):
        return {
            "count": 0.0,
            "bias": math.nan,
            "mae": math.nan,
            "median_absolute_error": math.nan,
            "median_relative_error": math.nan,
            "correlation": math.nan,
            "bland_altman_lower": math.nan,
            "bland_altman_upper": math.nan,
        }

    bias = float(np.mean(differences))
    difference_std = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0

    return {
        "count": float(len(differences)),
        "bias": bias,
        "mae": float(np.mean(absolute_errors)),
        "median_absolute_error": float(np.median(absolute_errors)),
        "median_relative_error": float(np.median(relative_errors)),
        "correlation": correlation(candidate, reference),
        "bland_altman_lower": bias - 1.96 * difference_std,
        "bland_altman_upper": bias + 1.96 * difference_std,
    }


def print_summary_line(name: str, summary: dict[str, float]) -> None:
    print(
        f"{name:<8} "
        f"n={int(summary['count']):>3}  "
        f"MAE={summary['mae']:.6g}  "
        f"median AE={summary['median_absolute_error']:.6g}  "
        f"median RE={summary['median_relative_error']:.2%}  "
        f"bias={summary['bias']:.6g}  "
        f"r={summary['correlation']:.4f}  "
        f"BA=[{summary['bland_altman_lower']:.6g}, {summary['bland_altman_upper']:.6g}]"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ParkinVox sustained-vowel features with Praat.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A sustained-vowel audio file or directory containing audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results.csv"),
        help="CSV output path (default: validation_results.csv).",
    )
    parser.add_argument("--f0-mae-hz", type=float, default=5.0)
    parser.add_argument("--jitter-median-relative-error", type=float, default=0.15)
    parser.add_argument("--shimmer-median-relative-error", type=float, default=0.15)
    parser.add_argument("--hnr-mae-db", type=float, default=2.0)
    parser.add_argument("--f1-mae-hz", type=float, default=50.0)
    parser.add_argument("--f2-mae-hz", type=float, default=100.0)
    parser.add_argument("--f3-mae-hz", type=float, default=150.0)
    parser.add_argument(
        "--allow-qc-failures",
        action="store_true",
        help="Include low-level, clipped, or short recordings in aggregate results.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    targets = AcceptanceTargets(
        f0_mae_hz=arguments.f0_mae_hz,
        jitter_median_relative_error=arguments.jitter_median_relative_error,
        shimmer_median_relative_error=arguments.shimmer_median_relative_error,
        hnr_mae_db=arguments.hnr_mae_db,
        f1_mae_hz=arguments.f1_mae_hz,
        f2_mae_hz=arguments.f2_mae_hz,
        f3_mae_hz=arguments.f3_mae_hz,
    )
    config = ExtractionConfig()
    audio_files = discover_audio_files(arguments.input)

    if not audio_files:
        print("No supported audio files found.", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    failures: list[tuple[Path, str]] = []

    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] {audio_path}")
        try:
            rows.append(compare_recording(audio_path, config))
        except Exception as error:
            failures.append((audio_path, f"{type(error).__name__}: {error}"))
            print(f"  ERROR: {type(error).__name__}: {error}")

    if not rows:
        print("No recordings could be analyzed.", file=sys.stderr)
        return 2

    results = pd.DataFrame(rows)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output, index=False)

    qc_failure_mask = (
        results["qc_low_level_flag"].astype(bool)
        | results["qc_clipping_flag"].astype(bool)
        | results["qc_short_recording_flag"].astype(bool)
    )

    if arguments.allow_qc_failures:
        aggregate_results = results
    else:
        aggregate_results = results.loc[~qc_failure_mask].copy()

    print()
    print(f"Analyzed: {len(results)}")
    print(f"QC failures: {int(qc_failure_mask.sum())}")
    print(f"Extraction failures: {len(failures)}")
    print(f"Included in aggregate comparison: {len(aggregate_results)}")
    print(f"CSV: {arguments.output.resolve()}")

    if failures:
        print("\nEXTRACTION FAILURES")
        for path, message in failures:
            print(f"- {path}: {message}")

    if aggregate_results.empty:
        print("\nNo QC-passing recordings remain for aggregate validation.")
        return 2

    metric_columns = {
        "F0": ("parkinvox_f0_median_hz", "praat_f0_median_hz"),
        "Jitter": ("parkinvox_jitter_local", "praat_jitter_local"),
        "Shimmer": ("parkinvox_shimmer_local", "praat_shimmer_local"),
        "HNR": ("parkinvox_hnr_db", "praat_hnr_db"),
        "F1": ("parkinvox_f1_median_hz", "praat_f1_median_hz"),
        "F2": ("parkinvox_f2_median_hz", "praat_f2_median_hz"),
        "F3": ("parkinvox_f3_median_hz", "praat_f3_median_hz"),
    }

    summaries = {
        name: summarize_metric(aggregate_results, name, candidate, reference)
        for name, (candidate, reference) in metric_columns.items()
    }

    print("\nAGGREGATE COMPARISON")
    for name, summary in summaries.items():
        print_summary_line(name, summary)

    checks = {
        "F0 MAE": summaries["F0"]["mae"] <= targets.f0_mae_hz,
        "Jitter median relative error": (
            summaries["Jitter"]["median_relative_error"]
            <= targets.jitter_median_relative_error
        ),
        "Shimmer median relative error": (
            summaries["Shimmer"]["median_relative_error"]
            <= targets.shimmer_median_relative_error
        ),
        "HNR MAE": summaries["HNR"]["mae"] <= targets.hnr_mae_db,
        "F1 MAE": summaries["F1"]["mae"] <= targets.f1_mae_hz,
        "F2 MAE": summaries["F2"]["mae"] <= targets.f2_mae_hz,
        "F3 MAE": summaries["F3"]["mae"] <= targets.f3_mae_hz,
    }

    print("\nACCEPTANCE CHECKS")
    for description, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {description}")

    if failures:
        print("FAIL: One or more recordings could not be analyzed")

    all_passed = all(checks.values()) and not failures
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
