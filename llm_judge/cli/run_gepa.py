import argparse

from dotenv import load_dotenv

from llm_judge import data as data_utils
from llm_judge import eval as eval_utils
from llm_judge import metrics, signatures
from llm_judge.optimizers import get_optimizer
from llm_judge.providers import setup_models


def parse_args():
    parser = argparse.ArgumentParser(description="Run GEPA optimization for clinical impact judge.")
    parser.add_argument("--data-path", type=str, default="llm_judge/dataset/primock_data_final_outcomes.csv", help="CSV file path.")
    parser.add_argument("--provider", type=str, default="openrouter", choices=["gemini", "bedrock", "openrouter"])
    parser.add_argument("--task-model", type=str, default="meta-llama/llama-3.3-70b-instruct")
    parser.add_argument("--reflection-model", type=str, default="anthropic/claude-4-sonnet")
    parser.add_argument("--no-separate-reflection", action="store_true", help="Use task model for reflection too.")
    parser.add_argument("--output", type=str, default="clinical_judge_gepa.json")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto", type=str, default="medium", help="GEPA auto level: light|medium|heavy")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not args.data_path:
        raise SystemExit("Please provide --data-path or set DATA_PATH.")

    separate_reflection = not args.no_separate_reflection
    print("=" * 80)
    print("DSPy Clinical Impact Judge - GEPA")
    print("=" * 80)

    # Data
    df = data_utils.load_dataset(args.data_path)
    trainset, valset, testset = data_utils.build_splits(
        df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    print(f"Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

    # Models
    reflection_model = args.reflection_model if separate_reflection else None
    task_lm, reflection_lm = setup_models(
        args.provider, task_model=args.task_model, reflection_model=reflection_model
    )
    print(f"Connected to provider={args.provider} task_model={args.task_model}")
    if reflection_lm:
        print(f"Using separate reflection model: {args.reflection_model}")

    # Optimizer
    optimizer = get_optimizer(
        "gepa",
        metric=metrics.gepa_feedback_metric,
        reflection_lm=reflection_lm,
        auto=args.auto,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        track_stats=True,
        seed=args.seed,
    )

    optimized_judge = optimizer.compile(
        signatures.ClinicalImpactJudge(),
        trainset=trainset,
        valset=valset,
    )
    optimized_judge.save(args.output)
    print(f"Saved optimized judge to {args.output}")

    # Evaluate on test
    eval_utils.evaluate_judge(optimized_judge, testset, name="GEPA Optimized")


if __name__ == "__main__":
    main()
