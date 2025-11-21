"""
Generate LLM-based alignment for evaluation purposes.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from aligner import ASRResult, GoldenUtterance, LLMTextAligner, OpenRouterClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_golden_transcript(golden_data):
    """Parse golden transcript from JSON file with transcript_golden_anonymized key."""
    transcript_text = golden_data.get("transcript_golden_anonymized", "")
    golden_utterances = []
    lines = transcript_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("["):
            continue
        try:
            timestamp_end = line.index("]")
            timestamp = line[1:timestamp_end]
            remaining = line[timestamp_end + 1 :].strip()
            if remaining.startswith("Patient:"):
                text = remaining[8:].strip()
                golden_utterances.append(
                    GoldenUtterance(text=text, timestamp=timestamp, sequence_order=len(golden_utterances))
                )
        except (ValueError, IndexError):
            logger.warning(f"Could not parse line: {line}")
            continue
    return golden_utterances


def parse_asr_results(asr_data):
    """Parse ASR results from JSON file."""
    asr_results = []
    for i, result in enumerate(asr_data):
        asr_results.append(
            ASRResult(
                id=result.get("id", f"asr_{i}"),
                text=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
                started_at=result.get("startedAt", ""),
                ended_at=result.get("endedAt", ""),
                received_at=result.get("receivedAt", ""),
                sequence_order=i,
            )
        )
    return asr_results


def convert_alignment_to_groundtruth_format(
    alignments, golden_utterances, asr_results, description: str = ""
):
    """Convert LLM alignment results to groundtruth format for comparison."""
    used_golden_indices = set()
    used_asr_indices = set()
    formatted_alignments = []

    for alignment in alignments:
        if alignment is None or alignment.match_type == "missing":
            continue

        if hasattr(alignment, "multi_golden_sequence") and alignment.multi_golden_sequence:
            golden_indices = alignment.multi_golden_sequence
        else:
            golden_indices = [alignment.golden.sequence_order]

        asr_indices = []
        asr_confidences = []
        if alignment.asr_fragments:
            asr_indices = [f.sequence_order for f in alignment.asr_fragments]
            asr_confidences = [f.confidence for f in alignment.asr_fragments]
        elif alignment.asr:
            asr_indices = [alignment.asr.sequence_order]
            asr_confidences = [alignment.asr.confidence]

        used_golden_indices.update(golden_indices)
        used_asr_indices.update(asr_indices)

        formatted_alignments.append(
            {
                "golden_indices": golden_indices,
                "golden_text": alignment.golden.text,
                "asr_indices": asr_indices,
                "asr_text": " ".join([asr_results[i].text for i in asr_indices]) if asr_indices else "",
                "asr_confidences": asr_confidences,
                "match_type": alignment.match_type,
                "notes": f"LLM generated with similarity score: {alignment.similarity_score:.3f}",
            }
        )

    all_golden_indices = set(range(len(golden_utterances)))
    all_asr_indices = set(range(len(asr_results)))
    unused_golden = all_golden_indices - used_golden_indices
    unused_asr = all_asr_indices - used_asr_indices

    unused_golden_results = [
        {"golden_index": idx, "golden_text": golden_utterances[idx].text} for idx in sorted(unused_golden)
    ]
    unused_asr_results = [{"asr_index": idx, "asr_text": asr_results[idx].text} for idx in sorted(unused_asr)]

    return {
        "description": description,
        "created_by": "llm_aligner",
        "total_golden_utterances": len(golden_utterances),
        "total_asr_results": len(asr_results),
        "unused_golden_results": unused_golden_results,
        "unused_asr_results": unused_asr_results,
        "alignments": formatted_alignments,
        "summary": {
            "total_alignments": len(formatted_alignments),
            "golden_coverage_percent": round(
                100 * len(used_golden_indices) / len(golden_utterances), 1
            )
            if golden_utterances
            else 0,
            "asr_usage_percent": round(100 * len(used_asr_indices) / len(asr_results), 1)
            if asr_results
            else 0,
        },
    }


def generate_alignment(golden_path: Path, asr_path: Path, output_path: Path):
    """Generate LLM alignment and save in groundtruth format."""
    logger.info(f"Loading golden transcript from: {golden_path}")
    with open(golden_path, "r") as f:
        golden_data = json.load(f)

    logger.info(f"Loading ASR results from: {asr_path}")
    with open(asr_path, "r") as f:
        asr_data = json.load(f)

    golden_utterances = parse_golden_transcript(golden_data)
    asr_results = parse_asr_results(asr_data)
    logger.info(f"Parsed {len(golden_utterances)} golden utterances and {len(asr_results)} ASR results")

    logger.info("Initializing LLM aligner with OpenRouter...")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    llm_client = OpenRouterClient(model=model)
    aligner = LLMTextAligner(llm_client)

    logger.info("Running LLM alignment...")
    alignments = aligner.align_utterances_with_llm(golden_utterances, asr_results)

    description = f"LLM-generated alignment for {golden_path.name} vs {asr_path.name}"
    formatted_alignment = convert_alignment_to_groundtruth_format(alignments, golden_utterances, asr_results, description)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted_alignment, f, indent=2)

    logger.info(f"Alignment saved to: {output_path}")
    logger.info(f"  Total alignments: {formatted_alignment['summary']['total_alignments']}")
    logger.info(f"  Golden coverage: {formatted_alignment['summary']['golden_coverage_percent']}%")
    logger.info(f"  ASR usage: {formatted_alignment['summary']['asr_usage_percent']}%")
    logger.info(f"  Unused golden: {len(formatted_alignment['unused_golden_results'])}")
    logger.info(f"  Unused ASR: {len(formatted_alignment['unused_asr_results'])}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM alignment for evaluation")
    parser.add_argument("--golden", "-g", required=True, help="Path to golden transcript JSON")
    parser.add_argument("--asr", "-a", required=True, help="Path to ASR results JSON")
    parser.add_argument("--output", "-o", required=True, help="Path to save alignment output")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    asr_path = Path(args.asr)
    output_path = Path(args.output)

    if not golden_path.exists():
        logger.error(f"Golden transcript not found: {golden_path}")
        sys.exit(1)
    if not asr_path.exists():
        logger.error(f"ASR results not found: {asr_path}")
        sys.exit(1)

    generate_alignment(golden_path, asr_path, output_path)


if __name__ == "__main__":
    main()
