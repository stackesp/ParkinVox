from pathlib import Path
from time import sleep

import requests
import soundfile as sf
import wfdb

BASE_URL = "https://physionet.org/files/voiced/1.0.0"
RAW_DIRECTORY = Path("voiced_raw")
OUTPUT_DIRECTORY = Path("validation_audio")
NUMBER_OF_ATTEMPTS = 5


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        return

    for attempt in range(NUMBER_OF_ATTEMPTS):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return
        except requests.RequestException:
            if attempt + 1 == NUMBER_OF_ATTEMPTS:
                raise

            sleep(10 * (attempt + 1))


def main() -> None:
    RAW_DIRECTORY.mkdir(exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    response = requests.get(f"{BASE_URL}/RECORDS?download", timeout=60)
    response.raise_for_status()
    record_names = response.text.splitlines()

    for index, record_name in enumerate(record_names, start=1):
        record_name = record_name.strip()

        if not record_name:
            continue

        for suffix in (".hea", ".dat", "-info.txt"):
            file_name = f"{record_name}{suffix}"
            download_file(
                f"{BASE_URL}/{file_name}?download",
                RAW_DIRECTORY / file_name,
            )

        output_path = OUTPUT_DIRECTORY / f"{record_name}.wav"

        if not output_path.exists():
            record = wfdb.rdrecord(str(RAW_DIRECTORY / record_name))
            sf.write(
                output_path,
                record.p_signal[:, 0],
                int(record.fs),
                subtype="FLOAT",
            )

        print(f"[{index}/{len(record_names)}] {output_path}")

    print(f"Converted {len(record_names)} recordings.")


if __name__ == "__main__":
    main()
