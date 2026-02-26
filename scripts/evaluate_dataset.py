"""Evaluate metrics over a CSV dataset.

Usage:
    python scripts/evaluate_dataset.py --csv data.csv --tier edit_distance_and_ngram
    python scripts/evaluate_dataset.py --csv data.csv --metrics wer,cer --output results.csv
    python scripts/evaluate_dataset.py --csv data.csv --judge
    python scripts/evaluate_dataset.py --csv data.csv --limit 10
    python scripts/evaluate_dataset.py --list-metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from metrics import calculate_metric, get_metric_info, list_metrics
from metrics.cleaning import get_clean_transcript
from metrics.model_cache import models
from metrics.registry import REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_model_based(name: str) -> bool:
    """True if the metric requires a model (has an optional-dep extra)."""
    return REGISTRY[name].extra is not None


def _default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_metrics.csv")


def _print_progress(prefix: str, current: int, total: int) -> None:
    print(f"\r  {prefix}: row {current}/{total}", end="", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Core metric evaluation
# ---------------------------------------------------------------------------


def evaluate_rows(
    df: pd.DataFrame,
    metric_names: list[str],
    gt_col: str,
    hyp_col: str,
    clean: bool,
    filter_nlts: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Evaluate metrics over a DataFrame, returning (results_df, none_counts).

    Uses a hybrid strategy:
    - Non-model metrics: single pass through rows (all metrics per row).
    - Model-based metrics: one metric at a time, scoring all rows, then
      clearing the model cache before the next metric. This prevents
      loading multiple large models simultaneously.
    """
    total = len(df)
    non_model = [n for n in metric_names if not _is_model_based(n)]
    model_based = [n for n in metric_names if _is_model_based(n)]

    results = pd.DataFrame(index=df.index, columns=metric_names, dtype=float)
    none_counts: dict[str, int] = {}

    # Pre-compute text as plain strings, coercing NaN -> empty string.
    gt_texts = df[gt_col].fillna("").astype(str)
    hyp_texts = df[hyp_col].fillna("").astype(str)

    if non_model:
        print(
            f"Processing {total} rows, {len(non_model)} Tier 1 metrics (row-by-row)...",
            file=sys.stderr,
        )
        for i, idx in enumerate(df.index):
            gt = gt_texts.at[idx]
            hyp = hyp_texts.at[idx]

            for name in non_model:
                results.at[idx, name] = _safe_score(
                    name, gt, hyp, clean, filter_nlts, none_counts
                )

            if (i + 1) % 50 == 0 or i + 1 == total:
                _print_progress("Row", i + 1, total)

        print(file=sys.stderr)

    if model_based:
        print(
            f"Processing {total} rows, {len(model_based)} model-based metrics "
            f"(metric-by-metric)...",
            file=sys.stderr,
        )
        for name in model_based:
            for i, idx in enumerate(df.index):
                gt = gt_texts.at[idx]
                hyp = hyp_texts.at[idx]
                results.at[idx, name] = _safe_score(
                    name, gt, hyp, clean, filter_nlts, none_counts
                )
                if (i + 1) % 50 == 0 or i + 1 == total:
                    _print_progress(name, i + 1, total)

            models.clear()
            print(file=sys.stderr)

    return results, none_counts


def _safe_score(
    name: str,
    gt: str,
    hyp: str,
    clean: bool,
    filter_nlts: bool,
    none_counts: dict[str, int],
) -> float | None:
    """Score a single metric. Returns None when the metric can't meaningfully score."""
    try:
        result = calculate_metric(name, gt, hyp, clean=clean, filter_nlts=filter_nlts)
    except Exception as exc:
        print(f"\n  Warning: {name} failed on row: {exc}", file=sys.stderr)
        result = None

    if result is None:
        none_counts[name] = none_counts.get(name, 0) + 1
    return result


# ---------------------------------------------------------------------------
# Judge evaluation
# ---------------------------------------------------------------------------


def evaluate_judge_rows(
    df: pd.DataFrame,
    artifact: str,
    provider: str,
    task_model: str,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Run the LLM judge on each row, returning (scores, reasonings, class_counts).

    Requires gt_context and hyp_context columns in the DataFrame.
    Returns Series aligned with df.index.
    """
    # Lazy imports -- only pulled in when --judge is used
    import dspy

    from llm_judge.metrics import parse_label
    from llm_judge.providers.factory import setup_models
    from llm_judge.signatures import ClinicalImpactJudge

    task_lm, _ = setup_models(provider, task_model=task_model, reflection_model=None)
    dspy.settings.configure(lm=task_lm)

    judge = ClinicalImpactJudge()
    judge.load(artifact)

    total = len(df)
    scores = pd.Series(index=df.index, dtype="Int64")  # nullable int
    reasonings = pd.Series(index=df.index, dtype=str)
    class_counts: dict[str, int] = {}

    gt_contexts = df["gt_context"].fillna("").astype(str)
    hyp_contexts = df["hyp_context"].fillna("").astype(str)

    print(f"Running judge on {total} rows...", file=sys.stderr)
    for i, idx in enumerate(df.index):
        gt_ctx = gt_contexts.at[idx]
        hyp_ctx = hyp_contexts.at[idx]

        try:
            prediction = judge(
                ground_truth_conversation=gt_ctx,
                transcription_conversation=hyp_ctx,
            )
            score = parse_label(prediction.clinical_impact)
            reasoning = getattr(prediction, "reasoning", "")

            if score is not None:
                scores.at[idx] = score
                key = str(score)
                class_counts[key] = class_counts.get(key, 0) + 1
            else:
                scores.at[idx] = pd.NA
                class_counts["parse_error"] = class_counts.get("parse_error", 0) + 1

            reasonings.at[idx] = reasoning
        except Exception as exc:
            print(f"\n  Warning: judge failed on row {idx}: {exc}", file=sys.stderr)
            scores.at[idx] = pd.NA
            reasonings.at[idx] = f"ERROR: {exc}"
            class_counts["error"] = class_counts.get("error", 0) + 1

        if (i + 1) % 10 == 0 or i + 1 == total:
            _print_progress("Judge", i + 1, total)

    print(file=sys.stderr)
    return scores, reasonings, class_counts


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_summary(
    df: pd.DataFrame,
    metric_names: list[str],
    none_counts: dict[str, int],
) -> None:
    """Print summary statistics to stdout."""
    print(f"\nResults ({len(df)} rows):")
    stats = df[metric_names].describe().loc[["mean", "std", "min", "max"]]
    print(f"  {'Metric':<22s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    for name in metric_names:
        print(
            f"  {name:<22s}"
            f" {stats.at['mean', name]:>8.4f}"
            f" {stats.at['std', name]:>8.4f}"
            f" {stats.at['min', name]:>8.4f}"
            f" {stats.at['max', name]:>8.4f}"
        )

    total_nones = sum(none_counts.values())
    if total_nones:
        print(f"\nNote: {total_nones} row(s) could not be scored (NaN in output):", file=sys.stderr)
        for name, count in none_counts.items():
            print(f"  {name}: {count}", file=sys.stderr)


def _print_judge_summary(class_counts: dict[str, int], total: int) -> None:
    """Print judge score distribution."""
    class_labels = {
        "0": "No impact",
        "1": "Minimal impact",
        "2": "Significant impact",
    }
    print("\nJudge results:")
    for key in ["0", "1", "2"]:
        count = class_counts.get(key, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  Class {key} ({class_labels[key]}): {count:>6d} ({pct:>5.1f}%)")

    for key in ["parse_error", "error"]:
        count = class_counts.get(key, 0)
        if count:
            print(f"  {key}: {count}")


def _print_list_metrics() -> None:
    """Print all registered metrics grouped by tier."""
    tiers = list_metrics()
    total = sum(len(names) for names in tiers.values())
    print(f"Available metrics ({total} total):\n")

    for tier, names in tiers.items():
        first_entry = REGISTRY[names[0]]
        extra_hint = ""
        if first_entry.extra:
            extra_hint = f"  [requires: uv sync --extra {first_entry.extra}]"

        print(f"  {tier} ({len(names)}):{extra_hint}")
        for name in names:
            info = get_metric_info(name)
            direction = "higher is better" if info.higher_is_better else "lower is better"
            print(f"    {name:<22s} {info.description:<42s} [{direction}]")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate metrics over a CSV dataset.",
    )
    parser.add_argument("--csv", type=Path, help="Input CSV path")
    parser.add_argument(
        "--gt-col", default="patient_ground_truth",
        help="GT column name (default: patient_ground_truth)",
    )
    parser.add_argument(
        "--hyp-col", default="patient_hypothesis",
        help="HYP column name (default: patient_hypothesis)",
    )

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--tier", type=str, help="Run all metrics in this tier")
    filter_group.add_argument(
        "--metrics", type=str, help="Comma-separated list of specific metrics to run"
    )

    parser.add_argument("--output", type=Path, help="Output CSV path (default: <input>_metrics.csv)")
    parser.add_argument("--limit", type=int, help="Only process first N rows")
    parser.add_argument(
        "--list-metrics", action="store_true", help="List all registered metrics and exit"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip transcript cleaning"
    )
    parser.add_argument(
        "--no-filter-nlts",
        action="store_true",
        help="Keep non-lexical tokens (uh, um, etc.) during cleaning",
    )

    # Judge flags
    parser.add_argument(
        "--judge", action="store_true",
        help="Run the LLM clinical impact judge (requires: uv sync --extra judge)",
    )
    parser.add_argument(
        "--artifact", type=str,
        default="llm_judge/results/clinical_judge_gepa.json",
        help="Path to saved judge artifact (default: llm_judge/results/clinical_judge_gepa.json)",
    )
    parser.add_argument(
        "--provider", type=str, default="openrouter",
        choices=["openrouter", "gemini", "bedrock"],
        help="LLM provider for the judge (default: openrouter)",
    )
    parser.add_argument(
        "--task-model", type=str,
        default="meta-llama/llama-3.3-70b-instruct:free",
        help="Model ID for the judge (default: meta-llama/llama-3.3-70b-instruct:free)",
    )

    args = parser.parse_args()

    if args.no_clean and args.no_filter_nlts:
        parser.error("--no-filter-nlts cannot be used with --no-clean (cleaning is already off)")

    if args.list_metrics:
        _print_list_metrics()
        sys.exit(0)

    if not args.csv:
        parser.error("--csv is required (unless --list-metrics)")

    csv_path: Path = args.csv
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.tier:
        tier_map = list_metrics()
        if args.tier not in tier_map:
            print(
                f"Error: unknown tier {args.tier!r}. Available: {sorted(tier_map.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        metric_names = tier_map[args.tier]
    elif args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",")]
        for name in metric_names:
            if name not in REGISTRY:
                print(f"Error: unknown metric {name!r}.", file=sys.stderr)
                sys.exit(1)
    else:
        metric_names = list(REGISTRY.keys())

    df = pd.read_csv(csv_path)

    if args.gt_col not in df.columns:
        print(f"Error: column {args.gt_col!r} not found in CSV.", file=sys.stderr)
        sys.exit(1)
    if args.hyp_col not in df.columns:
        print(f"Error: column {args.hyp_col!r} not found in CSV.", file=sys.stderr)
        sys.exit(1)

    if args.judge:
        missing = []
        if "gt_context" not in df.columns:
            missing.append("gt_context")
        if "hyp_context" not in df.columns:
            missing.append("hyp_context")
        if missing:
            print(
                f"Error: judge requires {', '.join(repr(c) for c in missing)} column(s) "
                f"in the CSV. These contain the conversation context for each utterance.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.limit:
        df = df.head(args.limit)

    # --- Metrics ---
    clean = not args.no_clean
    filter_nlts = not args.no_filter_nlts
    results_df, none_counts = evaluate_rows(
        df, metric_names, args.gt_col, args.hyp_col, clean, filter_nlts
    )

    if clean:
        df["clean_ground_truth"] = df[args.gt_col].fillna("").apply(
            lambda t: get_clean_transcript(str(t), remove_non_lexical_tokens=filter_nlts)
        )
        df["clean_hypothesis"] = df[args.hyp_col].fillna("").apply(
            lambda t: get_clean_transcript(str(t), remove_non_lexical_tokens=filter_nlts)
        )

    for name in metric_names:
        df[name] = results_df[name]

    _print_summary(df, metric_names, none_counts)

    # --- Judge ---
    if args.judge:
        scores, reasonings, class_counts = evaluate_judge_rows(
            df,
            artifact=args.artifact,
            provider=args.provider,
            task_model=args.task_model,
        )
        df["judge_clinical_impact"] = scores
        df["judge_reasoning"] = reasonings
        _print_judge_summary(class_counts, len(df))

    # --- Output ---
    output_path = args.output or _default_output_path(csv_path)
    df.to_csv(output_path, index=False)
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
