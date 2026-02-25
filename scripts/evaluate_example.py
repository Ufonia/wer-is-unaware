"""Evaluate metrics for a single ground-truth / hypothesis pair.

Usage:
    python scripts/evaluate_example.py --gt "the cat sat on the mat" --hyp "the cat sat on a mat"
    python scripts/evaluate_example.py --gt "..." --hyp "..." --metrics wer,bleu_1
    python scripts/evaluate_example.py --gt "..." --hyp "..." --tier edit_distance_and_ngram
    python scripts/evaluate_example.py --gt "..." --hyp "..." --judge --context-file context.txt
    python scripts/evaluate_example.py --list-metrics
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from metrics import calculate_metric, get_metric_info, list_metrics
from metrics.registry import REGISTRY


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_list_metrics() -> None:
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


def print_results(results: dict[str, float]) -> None:
    """Print metric results grouped by tier."""
    tier_results: dict[str, list[tuple[str, float]]] = {}
    for name, score in results.items():
        entry = REGISTRY[name]
        tier_results.setdefault(entry.tier, []).append((name, score))

    for tier, pairs in tier_results.items():
        print(f"\nMetrics ({tier}):")
        for name, score in pairs:
            print(f"  {name:<22s} {score:.4f}")


# ---------------------------------------------------------------------------
# Judge context helpers
# ---------------------------------------------------------------------------


def _load_context_file(path: str) -> str:
    """Read context file, stripping lines starting with #."""
    lines = Path(path).read_text().splitlines()
    content_lines = [line for line in lines if not line.startswith("#")]
    text = "\n".join(content_lines).strip()
    return text


def _next_index_from_context(context: str) -> int:
    """Parse the last (N) prefix from context, return N+1. Default 1 if none found."""
    matches = re.findall(r"\((\d+)\)", context)
    if matches:
        return int(matches[-1]) + 1
    return 1


def _build_example_context(
    gt: str, hyp: str, context_prefix: str | None
) -> tuple[str, str]:
    """Build GT/HYP conversation strings for the judge.

    context_prefix contains the common preceding turns (no final utterance).
    The final patient utterance is appended from gt/hyp respectively.
    If no context_prefix, just returns "Patient: <text>" for each.
    """
    if context_prefix:
        next_idx = _next_index_from_context(context_prefix)
        gt_ctx = f"{context_prefix}\n({next_idx}) Patient: {gt}"
        hyp_ctx = f"{context_prefix}\n({next_idx}) Patient: {hyp}"
    else:
        gt_ctx = f"Patient: {gt}"
        hyp_ctx = f"Patient: {hyp}"
    return gt_ctx, hyp_ctx


def _run_judge(
    gt_context: str,
    hyp_context: str,
    artifact: str,
    provider: str,
    task_model: str,
) -> None:
    """Load and run the LLM judge, printing results."""
    # Lazy imports — only pulled in when --judge is used
    import dspy

    from llm_judge.metrics import parse_label
    from llm_judge.providers.factory import setup_models
    from llm_judge.signatures import ClinicalImpactJudge

    task_lm, _ = setup_models(provider, task_model=task_model, reflection_model=None)
    dspy.settings.configure(lm=task_lm)

    judge = ClinicalImpactJudge()
    judge.load(artifact)

    prediction = judge(
        ground_truth_conversation=gt_context,
        transcription_conversation=hyp_context,
    )

    score = parse_label(prediction.clinical_impact)
    reasoning = getattr(prediction, "reasoning", "")

    class_labels = {
        0: "No impact",
        1: "Minimal impact",
        2: "Significant impact",
    }
    score_desc = class_labels.get(score, "Unknown") if score is not None else "Parse error"

    print("\nLLM Judge (Clinical Impact):")
    print(f"  Score:     {score} ({score_desc})")
    if reasoning:
        print(f"  Reasoning: {reasoning}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate metrics for a single ground-truth / hypothesis pair.",
    )
    parser.add_argument("--gt", type=str, help="Ground-truth transcript")
    parser.add_argument("--hyp", type=str, help="Hypothesis transcript")

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--tier", type=str, help="Run all metrics in this tier")
    filter_group.add_argument(
        "--metrics", type=str, help="Comma-separated list of specific metrics to run"
    )

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
    parser.add_argument(
        "--context-file", type=str,
        help="Path to a file with preceding conversation turns for the judge",
    )

    args = parser.parse_args()

    if args.no_clean and args.no_filter_nlts:
        parser.error("--no-filter-nlts cannot be used with --no-clean (cleaning is already off)")

    if args.context_file and not args.judge:
        parser.error("--context-file requires --judge")

    if args.list_metrics:
        print_list_metrics()
        sys.exit(0)

    if not args.gt or not args.hyp:
        parser.error("--gt and --hyp are required (unless --list-metrics)")

    # --- Metrics ---
    if args.tier:
        tier_metrics = list_metrics()
        if args.tier not in tier_metrics:
            print(
                f"Error: unknown tier {args.tier!r}. "
                f"Available: {sorted(tier_metrics.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        metric_names = tier_metrics[args.tier]
    elif args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",")]
        for name in metric_names:
            if name not in REGISTRY:
                print(f"Error: unknown metric {name!r}.", file=sys.stderr)
                sys.exit(1)
    else:
        metric_names = list(REGISTRY.keys())

    clean = not args.no_clean
    filter_nlts = not args.no_filter_nlts
    results: dict[str, float] = {}
    for name in metric_names:
        results[name] = calculate_metric(
            name, args.gt, args.hyp, clean=clean, filter_nlts=filter_nlts
        )

    print_results(results)

    # --- Judge ---
    if args.judge:
        context_prefix = None
        if args.context_file:
            context_prefix = _load_context_file(args.context_file)

        # Judge always receives uncleaned text
        gt_ctx, hyp_ctx = _build_example_context(args.gt, args.hyp, context_prefix)

        _run_judge(
            gt_context=gt_ctx,
            hyp_context=hyp_ctx,
            artifact=args.artifact,
            provider=args.provider,
            task_model=args.task_model,
        )


if __name__ == "__main__":
    main()
