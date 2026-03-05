# Data Preparation

This directory contains the data and scripts used to prepare the evaluation
dataset from the [PriMock57](https://github.com/babylonhealth/primock57)
corpus of simulated primary-care consultations.

## Dataset: PriMock57

**Paper:** Papadopoulos Korfiatis, Moramarco et al., *"PriMock57: A Dataset Of
Primary Care Mock Consultations"*
([arXiv:2204.00333](https://arxiv.org/abs/2204.00333))

**Repository:** <https://github.com/babylonhealth/primock57>

**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

We use **21 of the 57** consultations. The audio files are available from the
PriMock57 repository via
[git-lfs](https://github.com/babylonhealth/primock57/tree/main/audio).

### Attribution notice

The data in this directory is derived from the PriMock57 dataset, licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- **`data/textgrids/`** — Verbatim TextGrid transcript files from PriMock57
  (21 doctor + 21 patient, subset of the original 57 consultations).
- **`data/ground_truth/`** — Readable transcripts derived from the TextGrid
  files above (format conversion: TextGrid → `[MM:SS] Speaker: text`).
- **`data/asr_deepgram/`** — ASR hypothesis transcripts produced by running
  [Deepgram Nova-3](https://deepgram.com/) on PriMock57 patient-only audio tracks
  (adapted material).

### Consultations used

| # | Consultation ID |
|---|-----------------|
| 1 | `day1_consultation02` |
| 2 | `day1_consultation04` |
| 3 | `day1_consultation05` |
| 4 | `day1_consultation08` |
| 5 | `day1_consultation09` |
| 6 | `day1_consultation12` |
| 7 | `day1_consultation13` |
| 8 | `day1_consultation14` |
| 9 | `day2_consultation02` |
| 10 | `day2_consultation05` |
| 11 | `day2_consultation06` |
| 12 | `day3_consultation01` |
| 13 | `day3_consultation06` |
| 14 | `day3_consultation08` |
| 15 | `day4_consultation01` |
| 16 | `day4_consultation02` |
| 17 | `day4_consultation08` |
| 18 | `day5_consultation01` |
| 19 | `day5_consultation04` |
| 20 | `day5_consultation10` |
| 21 | `day5_consultation12` |

## Directory structure

```
data_preparation/
├── README.md
├── convert_textgrid.py        # TextGrid → readable .txt + structured JSON
├── downsample_audio.py        # 16 kHz → 8 kHz WAV downsampling
└── data/
    ├── textgrids/             # 42 TextGrid files (21 doctor + 21 patient)
    ├── ground_truth/          # 21 readable ground-truth transcripts
    └── asr_deepgram/          # 21 Deepgram ASR hypothesis JSONs
```

## Data pipeline

### 1. Source data

Each PriMock57 consultation provides:
- **Audio**: Separate patient and doctor WAV files at 16 kHz, 16-bit mono.
  We transcribe patient-only audio using streaming (real-time) ASR to match
  our production use case of live voice-agent transcription.
- **Transcripts**: TextGrid files with timing and speaker annotations

### 2. Ground-truth preparation

TextGrid transcripts are converted to a readable `[MM:SS] Speaker: text`
format. Consecutive utterances from the same speaker are merged, and TextGrid
markup tags (`<UNSURE>`, `<INAUDIBLE_SPEECH/>`, etc.) are stripped.

```
[00:00] Doctor: Hello?
[00:03] Patient: Hello. Can you hear me well?
[00:04] Doctor: Uh uh yes. I think. It's a bit better. ...
```

Pre-processed output is provided in `data/ground_truth/`. To regenerate from
the TextGrid files:

```bash
uv sync --extra data-prep
python data_preparation/convert_textgrid.py \
    --input-dir data_preparation/data/textgrids \
    --output-dir data_preparation/data/ground_truth
```

### 3. Audio downsampling (optional)

Our ASR system (Deepgram) expected 8 kHz input. If your ASR requires a
different sample rate, you can skip this step.

```bash
uv sync --extra data-prep
python data_preparation/downsample_audio.py \
    --input-dir /path/to/primock57/audio \
    --output-dir audio_8kHz/
```

The script validates WAV format (mono, 16-bit, expected sample rate) before
processing.

### 4. ASR transcription (not provided)

We used [Deepgram](https://deepgram.com/) for ASR. The resulting hypothesis
transcripts are in `data/asr_deepgram/` as JSON files. Users may substitute
any ASR system — the evaluation pipeline only needs reference and hypothesis
text.

### 5. Alignment and evaluation

Feed ground-truth and ASR transcripts into the `alignment/` pipeline, then
evaluate with the `metrics/` toolkit. See the
[main README](../README.md) for details.

## Citation

If you use PriMock57 data, please cite:

```bibtex
@inproceedings{korfiatis2022primock57,
  title={(in press): PriMock57: A Dataset Of Primary Care Mock Consultations},
  author={Papadopoulos Korfiatis, Alex and Moramarco, Francesco and Sarac, Radmila and Savkov, Aleksandar},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```
