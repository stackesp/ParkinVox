# ParkinVox

Voice feature extraction for sustained vowels, DDK, reading and spontaneous speech.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,validation]"
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -e ".[test,validation]"
```

## Test

```bash
pytest
```

## Use

```python
from parkinvox_src import extract_voice_features

features = extract_voice_features("recording.wav", task="sustained_vowel")
```

Supported tasks:

- `sustained_vowel`
- `ddk`
- `reading`
- `spontaneous`

## Validation

```bash
python validation/validate_praat.py validation_audio --output validation_results.csv
```

The validation script compares sustained-vowel measurements with Praat. It validates measurement agreement, not Parkinson's diagnosis.

## Data scripts

```bash
python scripts/download_vowels.py
python scripts/download_random_speech.py
```

Downloaded audio, datasets, validation output and trained models are ignored by Git.
