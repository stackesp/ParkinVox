from pathlib import Path

import soundfile as sf
from sbvoicedb import SbVoiceDb, Recording

NUMBER_OF_RECORDINGS = 20
OUTPUT_DIRECTORY = Path("validation_audio")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)

database = SbVoiceDb(
    "svd_cache",
    include_healthy=True,
    recording_filter=Recording.utterance == "a_n",
)

for index, recording in enumerate(database.iter_recordings()):
    sf.write(
        OUTPUT_DIRECTORY / f"vowel_{index + 1:02d}.wav",
        recording.nspdata,
        recording.rate,
    )

    if index + 1 == NUMBER_OF_RECORDINGS:
        break