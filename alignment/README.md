# Alignment Evaluation Toolkit

Lightweight, public-ready LLM sentence-level alignment and evaluation toolkit. Uses OpenRouter for LLM calls and uv for dependency management.

## Quickstart
- Install uv (https://github.com/astral-sh/uv) and run `uv sync`.
- Configure .env with `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`
- Generate an alignment:
  ```bash
  uv run python alignment/scripts/generate_alignment.py \
    --golden alignment/data/asr-transcripts/golden_sample.json \
    --asr alignment/data/asr-transcripts/patient_sample.json \
    --output alignment/data/llm-alignments/alignment_sample.json
  ```
- Evaluate against groundtruth:
  ```bash
  uv run python alignment/scripts/evaluate_alignment.py \
    --groundtruth alignment/data/groundtruth-alignments/groundtruth_alignment_sample.json \
    --llm alignment/data/llm-alignments/alignment_sample.json \
    --output alignment/results/alignment-evaluation/eval_sample.json
  ```
- Run both steps together:
  ```bash
  uv run python alignment/scripts/run_evaluation.py --case-id sample --asr-system demo
  ```

## Layout
- `aligner/` — core alignment logic (algorithmic + LLM-backed).
- `scripts/` — CLI wrappers for generate/evaluate/pipeline.
- `data/` — example inputs and placeholders.
- `results/` — evaluation outputs.

## Environment
- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (optional; defaults to `meta-llama/llama-3.3-70b-instruct:free`)
