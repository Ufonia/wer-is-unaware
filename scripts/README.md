# Evaluation Scripts

Two CLI scripts for computing ASR evaluation metrics. Both support the same metric registry — run `--list-metrics` on either to see what's available.

---

## `evaluate_example.py` — Single GT/HYP Pair

Evaluate one ground-truth / hypothesis pair against any combination of the 28 registered metrics.

### Flags

| Flag | Description |
|---|---|
| `--gt TEXT` | Ground-truth transcript (required unless `--list-metrics`) |
| `--hyp TEXT` | Hypothesis transcript (required unless `--list-metrics`) |
| `--tier TIER` | Run all metrics in this tier (mutually exclusive with `--metrics`) |
| `--metrics M1,M2,...` | Comma-separated list of specific metrics (mutually exclusive with `--tier`) |
| `--list-metrics` | List all registered metrics and exit |
| `--no-clean` | Skip transcript cleaning (numbers, punctuation, casing) |
| `--no-filter-nlts` | Keep non-lexical tokens ("uh", "um", etc.) during cleaning (mutually exclusive with `--no-clean`) |
| `--judge` | Run the LLM clinical impact judge (requires `uv sync --extra judge`) |
| `--artifact PATH` | Path to saved judge artifact (default: `llm_judge/results/clinical_judge_gepa.json`) |
| `--provider NAME` | LLM provider: `openrouter` (default), `gemini`, `bedrock` |
| `--task-model ID` | Model ID for the judge (default: `meta-llama/llama-3.3-70b-instruct:free`) |
| `--context-file PATH` | Path to a file with preceding conversation turns for the judge |

### Examples

**All installed metrics:**
```bash
python scripts/evaluate_example.py \
  --gt "I have been experiencing chest pain for three days" \
  --hyp "I have been experiencing chess pain for three days"
```

**Tier 1 only (edit-distance and n-gram):**
```bash
python scripts/evaluate_example.py \
  --gt "the pain is in my left arm" \
  --hyp "the pain is in my left arm" \
  --tier edit_distance_and_ngram
```

**Specific metrics:**
```bash
python scripts/evaluate_example.py \
  --gt "I take metformin twice daily" \
  --hyp "I take metformin twice daily" \
  --metrics wer,cer,bleu_1,sbert_similarity
```

**With the LLM judge:**
```bash
python scripts/evaluate_example.py \
  --gt "I have been taking aspirin for my heart" \
  --hyp "I have been taking aspirin for my head" \
  --judge --provider openrouter
```

**With a context file for the judge:**
```bash
python scripts/evaluate_example.py \
  --gt "yes twice a day" \
  --hyp "yes twice today" \
  --judge --context-file context.txt
```

The context file should contain the preceding conversation turns (plain text) so the judge can assess clinical impact in context.

---

## `evaluate_dataset.py` — Batch CSV Evaluation

Compute metrics over every row of a CSV file. Outputs a new CSV with the original columns plus metric scores and cleaned text columns.

### Flags

| Flag | Description |
|---|---|
| `--csv PATH` | Input CSV path (required unless `--list-metrics`) |
| `--gt-col NAME` | Ground-truth column name (default: `patient_ground_truth`) |
| `--hyp-col NAME` | Hypothesis column name (default: `patient_hypothesis`) |
| `--tier TIER` | Run all metrics in this tier (mutually exclusive with `--metrics`) |
| `--metrics M1,M2,...` | Comma-separated list of specific metrics (mutually exclusive with `--tier`) |
| `--output PATH` | Output CSV path (default: `<input_stem>_metrics.csv`) |
| `--limit N` | Only process the first N rows |
| `--list-metrics` | List all registered metrics and exit |
| `--no-clean` | Skip transcript cleaning |
| `--no-filter-nlts` | Keep non-lexical tokens during cleaning (mutually exclusive with `--no-clean`) |
| `--judge` | Run the LLM clinical impact judge (requires `uv sync --extra judge`) |
| `--artifact PATH` | Path to saved judge artifact (default: `llm_judge/results/clinical_judge_gepa.json`) |
| `--provider NAME` | LLM provider: `openrouter` (default), `gemini`, `bedrock` |
| `--task-model ID` | Model ID for the judge (default: `meta-llama/llama-3.3-70b-instruct:free`) |

### Output Format

The output CSV contains:
- All original columns from the input CSV
- `clean_ground_truth` / `clean_hypothesis` — cleaned text (unless `--no-clean`)
- One column per metric (column names match registry keys)
- `judge_clinical_impact` / `judge_reasoning` (if `--judge` was used)

### Examples

**Tier 1 on a CSV:**
```bash
python scripts/evaluate_dataset.py \
  --csv my_data.csv \
  --tier edit_distance_and_ngram
```

**Specific metrics:**
```bash
python scripts/evaluate_dataset.py \
  --csv my_data.csv \
  --metrics wer,cer,sbert_similarity \
  --output my_results.csv
```

**With the judge** (requires `gt_context` and `hyp_context` columns in the CSV):
```bash
python scripts/evaluate_dataset.py \
  --csv my_data.csv \
  --judge --provider openrouter
```

**Reproduce paper results** (using the shipped metrics CSV):
```bash
python scripts/evaluate_dataset.py \
  --csv metrics/data/primock_metrics_subset.csv \
  --gt-col clean_ground_truth \
  --hyp-col clean_hypothesis \
  --tier edit_distance_and_ngram \
  --no-clean
```

Note: `--no-clean` is used here because the shipped CSV already contains cleaned text in the `clean_*` columns.

---

## Judge Prerequisites

The `--judge` flag requires:

1. **Install judge dependencies:** `uv sync --extra judge`
2. **API key** — set up a `.env` file (see [`.env.example`](../.env.example)):
   - **OpenRouter** (default): set `OPENROUTER_API_KEY`
   - **Gemini**: set `GCP_PROJECT_ID` and `GCP_LOCATION`
   - **Bedrock**: set `AWS_REGION` (uses AWS credential chain)
3. **Judge artifact** — the optimized judge is shipped at `llm_judge/results/clinical_judge_gepa.json`. Use `--artifact` to point to a different one.

For full details on the judge (training, GEPA optimization, evaluation), see [`llm_judge/README.md`](../llm_judge/README.md).
