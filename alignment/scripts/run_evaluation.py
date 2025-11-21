import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation_pipeline(
    golden_file: str, asr_file: str, groundtruth_file: str, case_id: str, asr_system: str
):
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    golden_path = data_dir / "asr-transcripts" / golden_file
    asr_path = data_dir / "asr-transcripts" / asr_file
    groundtruth_path = data_dir / "groundtruth-alignments" / groundtruth_file

    llm_alignment_path = data_dir / "llm-alignments" / f"alignment_{case_id}_{asr_system}.json"
    eval_results_path = base_dir / "results" / "alignment-evaluation" / f"eval_{case_id}_{asr_system}.json"

    for path, name in [
        (golden_path, "Golden transcript"),
        (asr_path, "ASR results"),
        (groundtruth_path, "Groundtruth alignment"),
    ]:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            return False

    logger.info("\n" + "=" * 80)
    logger.info(f"Running Evaluation Pipeline for Case {case_id} ({asr_system})")
    logger.info("=" * 80 + "\n")

    logger.info("Step 1: Generating LLM alignment...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(base_dir / "scripts" / "generate_alignment.py"),
                "--golden",
                str(golden_path),
                "--asr",
                str(asr_path),
                "--output",
                str(llm_alignment_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate LLM alignment: {e}")
        logger.error(e.stderr)
        return False

    logger.info("\nStep 2: Evaluating alignment against groundtruth...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(base_dir / "scripts" / "evaluate_alignment.py"),
                "--groundtruth",
                str(groundtruth_path),
                "--llm",
                str(llm_alignment_path),
                "--output",
                str(eval_results_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to evaluate alignment: {e}")
        logger.error(e.stderr)
        return False

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    logger.info(f"  LLM Alignment: {llm_alignment_path}")
    logger.info(f"  Evaluation Results: {eval_results_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run alignment evaluation pipeline")
    parser.add_argument("--case-id", "-c", required=True, help="Test case ID (e.g., 'sample')")
    parser.add_argument("--asr-system", "-s", required=True, help="ASR system name (e.g., 'demo')")
    parser.add_argument(
        "--golden",
        "-g",
        help="Golden transcript filename (default: golden_{case_id}.json)",
    )
    parser.add_argument(
        "--asr",
        "-a",
        help="ASR results filename (default: patient_{case_id}_{asr_system}.json)",
    )
    parser.add_argument(
        "--groundtruth",
        "-t",
        help="Groundtruth alignment filename (default: groundtruth_alignment_{case_id}_{asr_system}.json)",
    )
    args = parser.parse_args()

    golden_file = args.golden or f"golden_{args.case_id}.json"
    asr_file = args.asr or f"patient_{args.case_id}_{args.asr_system}.json"
    groundtruth_file = args.groundtruth or f"groundtruth_alignment_{args.case_id}_{args.asr_system}.json"

    success = run_evaluation_pipeline(golden_file, asr_file, groundtruth_file, args.case_id, args.asr_system)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
