# llm_judge quickstart

Steps to run the clinical impact judge with GEPA or MIPROv2.

## Prereqs
- Data CSV with columns: `gt_context`, `hyp_context`, `final_outcome` (labels 0/1/2).
- Bundled dataset: `llm_judge/dataset/primock_data_final_outcomes.csv`

## Included optimized judges
- GEPA: `llm_judge/results/clinical_judge_gepa.json`
- MIPROv2: `llm_judge/results/clinical_judge_mipro_v2.json`

## Provider setup
- **OpenRouter (default):** set `OPENROUTER_API_KEY` (and optional `OPENROUTER_BASE_URL`); choose model e.g. `anthropic/claude-3.5-sonnet`.
- **Gemini:** set `GCP_PROJECT_ID`, `GCP_LOCATION` (optional, defaults to `us-central1`); choose model e.g. `gemini-2.5-pro`.
- **Bedrock:** set `AWS_REGION`; choose models e.g. `us.meta.llama3-3-70b-instruct-v1:0`.

## Run GEPA optimization
```bash
python -m llm_judge.cli.run_gepa \
  --data-path llm_judge/dataset/primock_data_final_outcomes.csv \
  --provider openrouter \
  --task-model meta-llama/llama-3.3-70b-instruct \
  --reflection-model anthropic/claude-4-sonnet \
  --output llm_judge/results/clinical_judge_gepa.json
```
Flags of note: `--no-separate-reflection` to reuse the task model; `--auto` for GEPA intensity (`light|medium|heavy`).

## Run MIPROv2 optimization
```bash
python -m llm_judge.cli.run_mipro \
  --data-path llm_judge/dataset/primock_data_final_outcomes.csv \
  --provider openrouter \
  --task-model anthropic/claude-3.5-sonnet \
  --output llm_judge/results/clinical_judge_mipro_v2.json
```
Flags of note: `--auto` for MIPRO intensity (`light|medium|heavy`).

## Evaluate a saved judge
```bash
python -m llm_judge.cli.run_eval \
  --artifact llm_judge/results/clinical_judge_gepa.json \
  --data-path llm_judge/dataset/primock_data_final_outcomes.csv
```

## Splits
Default splits match the original script: test=50, val=30, remaining for train. Override with `--test-size` / `--val-size`. Seed defaults to 42.

## Using the judge via evaluation scripts

The judge can also be invoked through the general-purpose evaluation scripts in `scripts/`. This is useful for running metrics and the judge together in a single pass.

### Single pair

```bash
# Bare mode (no context — judge sees "Patient: <text>" only)
python scripts/evaluate_example.py \
  --gt "it's clearing up" \
  --hyp "it's clear but" \
  --metrics wer \
  --judge

# With preceding conversation context
python scripts/evaluate_example.py \
  --gt "it's clearing up" \
  --hyp "it's clear but" \
  --metrics wer \
  --judge \
  --context-file llm_judge/context_example.txt
```

The `--context-file` provides preceding conversation turns. The final patient utterance is appended automatically from `--gt`/`--hyp`. See `llm_judge/context_example.txt` for the expected format.

### Dataset

```bash
python scripts/evaluate_dataset.py \
  --csv llm_judge/dataset/primock_data_final_outcomes.csv \
  --tier edit_distance_and_ngram \
  --judge
```

The dataset script reads pre-computed `gt_context` and `hyp_context` columns from the CSV. If these columns are missing, the script errors with a clear message. Output CSV will include `judge_clinical_impact` and `judge_reasoning` columns.

### Judge flags (both scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--judge` | off | Enable the LLM judge |
| `--artifact PATH` | `llm_judge/results/clinical_judge_gepa.json` | Path to saved judge artifact |
| `--provider` | `openrouter` | LLM provider (openrouter/gemini/bedrock) |
| `--task-model` | `meta-llama/llama-3.3-70b-instruct:free` | Model ID |
| `--context-file PATH` | — | Preceding turns file (evaluate_example.py only) |

The judge always receives **uncleaned** text (it was trained on raw utterances, not cleaned ones).
