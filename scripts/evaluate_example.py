"""Evaluate metrics for a single ground-truth / hypothesis pair.

Usage:
    python scripts/evaluate_example.py --gt "the cat sat on the mat" --hyp "the cat sat on a mat"
    python scripts/evaluate_example.py --gt "..." --hyp "..." --metrics wer,bleu_1
    python scripts/evaluate_example.py --gt "..." --hyp "..." --tier edit_distance_and_ngram
    python scripts/evaluate_example.py --list-metrics
"""

from __future__ import annotations

import argparse
import sys

from metrics import calculate_metric, get_metric_info, list_metrics
from metrics.registry import REGISTRY


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

    args = parser.parse_args()

    if args.no_clean and args.no_filter_nlts:
        parser.error("--no-filter-nlts cannot be used with --no-clean (cleaning is already off)")

    if args.list_metrics:
        print_list_metrics()
        sys.exit(0)

    if not args.gt or not args.hyp:
        parser.error("--gt and --hyp are required (unless --list-metrics)")

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


if __name__ == "__main__":
    main()
