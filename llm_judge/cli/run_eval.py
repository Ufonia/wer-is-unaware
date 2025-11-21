import argparse
import os

import dspy
from dotenv import load_dotenv

from llm_judge import data as data_utils
from llm_judge import eval as eval_utils
from llm_judge import signatures
from llm_judge.providers import setup_models


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved clinical judge artifact.")
    parser.add_argument("--artifact", type=str, required=True, help="Path to saved judge JSON.")
    parser.add_argument("--data-path", type=str, default=os.getenv("DATA_PATH"), help="CSV file path.")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", type=str, default="openrouter", choices=["gemini", "bedrock", "openrouter"])
    parser.add_argument("--task-model", type=str, default="meta-llama/llama-3.3-70b-instruct:free")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not args.data_path:
        raise SystemExit("Please provide --data-path or set DATA_PATH.")

    print("=" * 80)
    print("DSPy Clinical Impact Judge - Evaluation")
    print("=" * 80)

    task_lm, _ = setup_models(
        args.provider, task_model=args.task_model, reflection_model=None
    )

    dspy.settings.configure(lm=task_lm)

    df = data_utils.load_dataset(args.data_path)
    _, _, testset = data_utils.build_splits(
        df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )

    judge = signatures.ClinicalImpactJudge()
    judge.load(args.artifact)
    eval_utils.evaluate_judge(judge, testset[:10], name=os.path.basename(args.artifact))


if __name__ == "__main__":
    main()
