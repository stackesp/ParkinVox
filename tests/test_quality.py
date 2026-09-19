from pathlib import Path
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "src"
TEST_AUDIO_DIRECTORY = PROJECT_DIRECTORY / "tests" / "data"
sys.path.insert(0, str(SOURCE_DIRECTORY))

"""
EN: Run quality checks against the included example audio files. Just a simple smoke test.
TR: Dahil edilen örnek ses dosyalarına karşı kalite kontrollerini çalıştırır. Sadece basit bir smoke testi.
"""

from parkinvox_src.audio import load_audio
from parkinvox_src.audio import remove_dc_offset
from parkinvox_src.config import ExtractionConfig
from parkinvox_src.quality import calculate_quality_features

config = ExtractionConfig()

def inspect_audio(file_name: str) -> dict:
    audio_path = TEST_AUDIO_DIRECTORY / file_name

    audio, sample_rate, metadata = load_audio(
        audio_path
    )

    quality_features = calculate_quality_features(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
    )

    print(f"\n{file_name}")

    for metadata_name in sorted(metadata):
        metadata_value = metadata[metadata_name]
        print(f"  {metadata_name}: {metadata_value}")

    for feature_name in sorted(quality_features):
        feature_value = quality_features[feature_name]
        print(f"  {feature_name}: {feature_value}")

    return {
        "audio": audio,
        "metadata": metadata,
        "quality": quality_features,
    }


def check_result(
    description: str,
    condition: bool,
) -> bool:
    if condition:
        status = "PASS"
    else:
        status = "FAIL"

    print(f"{status}: {description}")
    return condition


def main() -> None:
    results = {
        "clean": inspect_audio("clean_sine.wav"),
        "silence": inspect_audio("silence.wav"),
        "clipped": inspect_audio("clipped_sine.wav"),
        "short": inspect_audio("short_sine.wav"),
        "quiet": inspect_audio("quiet_sine.wav"),
        "dc_offset": inspect_audio("dc_offset.wav"),
    }

    print("\nTEST RESULTS")

    checks = []

    """
    EN: The checks are located here.
    TR: Kontroller buraya yerleştirilmiştir.
    """

    clean_quality = results["clean"]["quality"]
    checks.append(check_result(
        "Clean audio is not marked as clipped",
        clean_quality["qc_clipping_flag"] is False,
    ))
    checks.append(check_result(
        "Clean audio is not marked as short",
        clean_quality["qc_short_recording_flag"] is False,
    ))

    silence_quality = results["silence"]["quality"]
    checks.append(check_result(
        "Silence is marked as low level",
        silence_quality["qc_low_level_flag"] is True,
    ))
    checks.append(check_result(
        "Silence fraction is close to 100 percent",
        silence_quality["qc_silence_fraction"] > 0.95,
    ))

    clipped_quality = results["clipped"]["quality"]
    checks.append(check_result(
        "Clipped audio is marked as clipped",
        clipped_quality["qc_clipping_flag"] is True,
    ))

    short_quality = results["short"]["quality"]
    checks.append(check_result(
        "Short audio is marked as short",
        short_quality["qc_short_recording_flag"] is True,
    ))

    quiet_quality = results["quiet"]["quality"]
    checks.append(check_result(
        "Quiet audio is marked as low level",
        quiet_quality["qc_low_level_flag"] is True,
    ))

    dc_offset_audio = results["dc_offset"]["audio"]
    centered_audio = remove_dc_offset(
        dc_offset_audio
    )
    centered_average = centered_audio.mean()
    checks.append(check_result(
        "DC-offset removal centers the waveform",
        abs(centered_average) < 0.000001,
    ))

    number_of_tests = len(checks)
    number_of_passed_tests = sum(checks)

    print(
        f"\n{number_of_passed_tests}/{number_of_tests} tests passed"
    )

    if number_of_passed_tests != number_of_tests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
