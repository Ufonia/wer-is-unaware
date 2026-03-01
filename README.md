# WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient-Facing Dialogue

This repository hosts the code, models, and datasets accompanying the paper. The work investigates how Automatic Speech Recognition (ASR) errors **distort clinical meaning in patient-facing dialogue** — and shows that traditional metrics like Word Error Rate (WER) fail to capture real clinical risk. The project includes scripts for aligning ground-truth utterances to ASR-generated utterances using an **LLM-based semantic aligner**, and optimizing an **LLM-as-a-Judge for clinical impact assessment** using GEPA through DSPy.

## Abstract
![WER is Unaware Overview](overview.png)

As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's kappa of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue.

## What This Repo Provides

- **28 ASR evaluation metrics** across 3 tiers — from simple edit-distance metrics (WER, CER) through n-gram overlap (BLEU, ROUGE, METEOR) to learned semantic metrics (BERTScore, SeMaScore, Intelligibility, Heval)
- **LLM-based clinical impact judge** — GEPA-optimized judge that replicates expert clinician assessment of ASR error severity
- **Two CLI evaluation scripts** — evaluate a single GT/HYP pair (`evaluate_example.py`) or batch-process a CSV (`evaluate_dataset.py`). See [`scripts/README.md`](scripts/README.md) for full usage documentation.

## Quick Start

```bash
git clone https://github.com/YOUR_ORG/wer-is-unaware.git && cd wer-is-unaware
uv sync                          # Tier 1 metrics (15 metrics, no GPU)

# Evaluate a single pair
python scripts/evaluate_example.py \
  --gt "I have been experiencing chest pain for three days" \
  --hyp "I have been experiencing chess pain for three days"

# List all available metrics
python scripts/evaluate_example.py --list-metrics
```

## Dependency Tiers

| Install Command | What You Get | Notes |
|---|---|---|
| `uv sync` | Tier 1: WER, CER, BLEU, ROUGE, etc. (15 metrics) | No GPU needed |
| `uv sync --extra learned-semantic` | + Tiers 2 & 3 (13 more metrics) | Models auto-download on first run (~10 GB) |
| `uv sync --extra bleurt` | + BLEURT & Clinical BLEURT | TensorFlow required, manual checkpoint download |
| `uv sync --extra judge` | LLM clinical impact judge | Requires API key (`.env`) |
| `uv sync --extra all-metrics` | All 28 metrics | |
| `uv sync --extra all` | Everything (metrics + judge + data-prep + plot + dev) | |

After install, download required NLTK data (needed for BLEU, METEOR):
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"
```

## Manual Downloads

Learned-semantic models (Tiers 2 & 3) auto-download from HuggingFace on first use (~10 GB total, cached in `~/.cache/huggingface/`). Two metrics require manual checkpoint downloads:

1. **BARTScore (ParaBank2)** — download the `.pth` checkpoint from [neulab/BARTScore](https://github.com/neulab/BARTScore),[checkpoint google-drive](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view), then set in `.env`:
   ```
   BARTSCORE_CHECKPOINT=path/to/bart_score.pth
   ```

2. **Clinical BLEURT** — download the checkpoint following [ClinicalBLEURT instructions](https://github.com/abachaa/EvaluationMetrics-ACL23/blob/main/ClinicalBLEURT/README.md),[checkpoint google-drive](https://drive.google.com/file/d/1pjd8TxqGeYVdCWZHdXow8pDXpwRebwFY/view?usp=sharing) then set in `.env`:
   ```
   CLINICAL_BLEURT_CHECKPOINT=path/to/ClinicalBLEURT
   ```

Standard BLEURT uses the test checkpoint bundled with the package (no manual download needed). See [`.env.example`](.env.example) for all environment variable names.

## Folder Structure

- `metrics/` — ASR evaluation metrics toolkit (28 metrics across 3 tiers). API: `from metrics import calculate_metric, list_metrics`.
- `scripts/` — CLI evaluation scripts for single-pair and batch CSV evaluation. See [`scripts/README.md`](scripts/README.md).
- `alignment/` — semantic alignment toolkit (aligner code, scripts, sample data, sample results). See [`alignment/README.md`](alignment/README.md).
- `llm_judge/` — clinical impact judge (signatures, metrics, providers, optimizers, CLI, bundled dataset, saved judges). See [`llm_judge/README.md`](llm_judge/README.md).
- `data_preparation/` — data pipeline scripts and PriMock57 transcript data. See [`data_preparation/README.md`](data_preparation/README.md).
- `tests/` — test suite (`pytest tests/`).

## Dataset

Our evaluation uses 21 consultations from the [PriMock57](https://github.com/babylonhealth/primock57) dataset of simulated primary-care consultations (CC BY 4.0). Ground-truth transcripts, ASR hypotheses, and preparation scripts are in [`data_preparation/`](data_preparation/README.md).

> Papadopoulos Korfiatis, A., Moramarco, F., Sarac, R. & Savkov, A. (2022).
> *PriMock57: A Dataset Of Primary Care Mock Consultations.* ACL 2022.

The repo ships two CSVs:

| File | Rows | Description |
|---|---|---|
| `llm_judge/dataset/primock_data_final_outcomes.csv` | 175 | Clinician-annotated clinical-impact labels (judge dataset) |
| `metrics/data/primock_metrics_subset.csv` | 157 | Pre-computed metric scores for utterances with non-zero WER after NLT filtering |

### Column Reference

**Shared base columns** (19 columns, present in both CSVs):

| Column | Description |
|---|---|
| `composite_key` | Unique utterance identifier (`call_id` + `interaction_index`) |
| `dora_or_primock` | Dataset source identifier |
| `interaction_index` | Turn index within the consultation |
| `call_id` | Consultation identifier |
| `doctor` | Doctor identifier |
| `patient_ground_truth` | Original ground-truth patient utterance |
| `patient_hypothesis` | ASR-generated hypothesis |
| `alignment_similarity_score` | Semantic similarity score from the LLM aligner |
| `alignment_status` | Alignment classification (e.g. match, mismatch) |
| `provider` | ASR system provider |
| `clinician_a` | Clinician A's impact label |
| `justification_a` | Clinician A's justification |
| `clinician_b` | Clinician B's impact label |
| `justification_b` | Clinician B's justification |
| `resolved_label` | Resolved label after adjudication |
| `disagreement_final_reasoning_from_meeting` | Resolution reasoning (if clinicians disagreed) |
| `final_outcome` | Final clinical impact label (0 = No, 1 = Minimal, 2 = Significant) |

**Judge CSV only:**

| Column | Description |
|---|---|
| `norm_ground_truth` | Cleaned text WITHOUT NLT filtering (fillers like "uh", "um" preserved) |
| `norm_hypothesis` | Cleaned hypothesis WITHOUT NLT filtering |
| `gt_context` | Preceding conversation context (ground-truth side) |
| `hyp_context` | Preceding conversation context (hypothesis side) |

**Metrics CSV only:**

| Column | Description |
|---|---|
| `gt_context` | Preceding conversation context (ground-truth side) |
| `hyp_context` | Preceding conversation context (hypothesis side) |
| `clean_ground_truth` | Cleaned text WITH NLT filtering (fillers removed) |
| `clean_hypothesis` | Cleaned hypothesis WITH NLT filtering |
| 28 metric columns | Pre-computed scores (names match registry keys — run `python scripts/evaluate_example.py --list-metrics` to list) |

**Why two text columns?** The judge CSV uses `norm_*` text (non-lexical tokens like "uh", "um" preserved) because these are needed for utterance filtering. The metrics CSV uses `clean_*` text (NLTs filtered out) because metrics are computed on cleaned text. Some utterances appear in the judge CSV but not the metrics CSV because they had zero WER after NLT filtering and were excluded from the metrics dataset.

## Paper

Preprint available on arXiv: https://arxiv.org/abs/2511.16544

## Citation

```bibtex
@misc{ellis2025werunawareassessingasr,
      title={WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue},
      author={Zachary Ellis and Jared Joselowitz and Yash Deo and Yajie He and Anna Kalygina and Aisling Higham and Mana Rahimzadeh and Yan Jia and Ibrahim Habli and Ernest Lim},
      year={2025},
      eprint={2511.16544},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.16544},
}
```
