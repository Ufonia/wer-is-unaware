import argparse
import os

from dotenv import load_dotenv

from llm_judge import data as data_utils
from llm_judge import eval as eval_utils
from llm_judge import metrics, signatures
from llm_judge.optimizers import get_optimizer
from llm_judge.providers import setup_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIPROv2 optimization for clinical impact judge.")
    parser.add_argument("--data-path", type=str, default=os.getenv("DATA_PATH"), help="CSV file path.")
    parser.add_argument("--provider", type=str, default="openrouter", choices=["gemini", "bedrock", "openrouter"])
    parser.add_argument("--task-model", type=str, default="anthropic/claude-3.5-sonnet")
    parser.add_argument("--output", type=str, default="clinical_judge_mipro.json")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto", type=str, default="medium", help="MIPRO auto level: light|medium|heavy")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not args.data_path:
        raise SystemExit("Please provide --data-path or set DATA_PATH.")

    print("=" * 80)
    print("DSPy Clinical Impact Judge - MIPROv2")
    print("=" * 80)

    # Data
    df = data_utils.load_dataset(args.data_path)
    trainset, valset, testset = data_utils.build_splits(
        df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

    # Models
    task_lm, _ = setup_models(args.provider, task_model=args.task_model)
    print(f"✓ Connected to provider={args.provider} task_model={args.task_model}")

    # Optimizer
    optimizer = get_optimizer(
        "mipro",
        metric=metrics.simple_metric,
        auto=args.auto,
        seed=args.seed,
    )

    optimized_judge = optimizer.compile(
        signatures.ClinicalImpactJudge(),
        trainset=trainset,
        requires_permission_to_run=False,
    )
    optimized_judge.save(args.output)
    print(f"✓ Saved optimized judge to {args.output}")

    # Evaluate on test
    eval_utils.evaluate_judge(optimized_judge, testset, name="MIPROv2 Optimized")


if __name__ == "__main__":
    main()
