from pathlib import Path

import soundfile as sf
from datasets import load_dataset


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
TEST_AUDIO_DIRECTORY = PROJECT_DIRECTORY / "test_audio"


def main() -> None:
    TEST_AUDIO_DIRECTORY.mkdir(exist_ok=True)

    print("Getting random speech samples...")

    dataset = load_dataset("PolyAI/minds14", "en-US", split="train", streaming=True)

    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    for index, sample in enumerate(dataset.take(2)):
        audio = sample["audio"]

        output_path = TEST_AUDIO_DIRECTORY / f"random_speech_{index + 1}.wav"

        sf.write(output_path, audio["array"], audio["sampling_rate"])

        print(f"Saved: {output_path}")
        print(f"Text: {sample.get('transcription', '')}")

    print("Done.")


if __name__ == "__main__":
    main()