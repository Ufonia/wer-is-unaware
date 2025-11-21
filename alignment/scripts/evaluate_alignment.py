"""
Evaluate LLM-based transcript aligner against groundtruth alignments.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AlignmentMetrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def to_dict(self) -> Dict:
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


def extract_unmatched_indices(alignment_data: Dict) -> Tuple[Set[int], Set[int]]:
    unmatched_golden = set()
    unmatched_asr = set()

    if "unused_golden_results" in alignment_data:
        for item in alignment_data["unused_golden_results"]:
            unmatched_golden.add(item["golden_index"])
    if "unused_asr_results" in alignment_data:
        for item in alignment_data["unused_asr_results"]:
            unmatched_asr.add(item["asr_index"])

    if "unmatched_golden_utterances" in alignment_data:
        for item in alignment_data["unmatched_golden_utterances"]:
            unmatched_golden.add(item["golden_index"])
    if "unmatched_asr_results" in alignment_data:
        for item in alignment_data["unmatched_asr_results"]:
            unmatched_asr.add(item["asr_index"])

    aligned_golden = set()
    aligned_asr = set()
    if "alignments" in alignment_data:
        for alignment in alignment_data["alignments"]:
            if "golden_indices" in alignment:
                golden_indices = alignment["golden_indices"]
                if isinstance(golden_indices, list):
                    aligned_golden.update(golden_indices)
                else:
                    aligned_golden.add(golden_indices)
            if "asr_indices" in alignment:
                asr_indices = alignment["asr_indices"]
                if isinstance(asr_indices, list):
                    aligned_asr.update(asr_indices)
                else:
                    aligned_asr.add(asr_indices)

    return unmatched_golden, unmatched_asr


def get_all_indices(alignment_data: Dict) -> Tuple[Set[int], Set[int]]:
    total_golden = alignment_data.get("total_golden_utterances", 0)
    total_asr = alignment_data.get("total_asr_results", 0)
    return set(range(total_golden)), set(range(total_asr))


def check_duplicate_usage(alignment_data: Dict) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    from collections import Counter

    all_asr = []
    all_golden = []

    if "alignments" in alignment_data:
        for alignment in alignment_data["alignments"]:
            if "asr_indices" in alignment:
                asr_indices = alignment["asr_indices"]
                if isinstance(asr_indices, list):
                    all_asr.extend(asr_indices)
                else:
                    all_asr.append(asr_indices)
            if "golden_indices" in alignment:
                golden_indices = alignment["golden_indices"]
                if isinstance(golden_indices, list):
                    all_golden.extend(golden_indices)
                else:
                    all_golden.append(golden_indices)

    asr_counts = Counter(all_asr)
    golden_counts = Counter(all_golden)
    duplicate_asr = [(idx, count) for idx, count in asr_counts.items() if count > 1]
    duplicate_golden = [(idx, count) for idx, count in golden_counts.items() if count > 1]
    return sorted(duplicate_asr), sorted(duplicate_golden)


def build_alignment_map(alignment_data: Dict) -> Dict[Tuple[int, ...], List[int]]:
    alignment_map = {}
    if "alignments" in alignment_data:
        for alignment in alignment_data["alignments"]:
            golden_indices = alignment.get("golden_indices", [])
            asr_indices = alignment.get("asr_indices", [])
            if not isinstance(golden_indices, list):
                golden_indices = [golden_indices]
            if not isinstance(asr_indices, list):
                asr_indices = [asr_indices]
            golden_key = tuple(sorted(golden_indices))
            alignment_map[golden_key] = sorted(asr_indices)
    return alignment_map


def compare_alignment_structures(gt_data: Dict, llm_data: Dict) -> List[Dict]:
    mismatches = []
    gt_map = build_alignment_map(gt_data)
    llm_map = build_alignment_map(llm_data)

    for golden_key, gt_asr_indices in gt_map.items():
        llm_asr_indices = llm_map.get(golden_key)
        if llm_asr_indices is None:
            mismatches.append(
                {"golden_indices": list(golden_key), "gt_asr_indices": gt_asr_indices, "llm_asr_indices": []}
            )
        elif gt_asr_indices != llm_asr_indices:
            mismatches.append(
                {
                    "golden_indices": list(golden_key),
                    "gt_asr_indices": gt_asr_indices,
                    "llm_asr_indices": llm_asr_indices,
                }
            )

    return mismatches


def evaluate_alignment(groundtruth_path: Path, llm_path: Path, output_path: Path) -> Dict:
    with open(groundtruth_path, "r") as f:
        gt_data = json.load(f)
    with open(llm_path, "r") as f:
        llm_data = json.load(f)

    # Unmatched detection metrics
    gt_unmatched_golden, gt_unmatched_asr = extract_unmatched_indices(gt_data)
    llm_unmatched_golden, llm_unmatched_asr = extract_unmatched_indices(llm_data)
    all_golden, all_asr = get_all_indices(gt_data)

    golden_metrics = AlignmentMetrics(
        tp=len(gt_unmatched_golden & llm_unmatched_golden),
        tn=len((all_golden - gt_unmatched_golden) & (all_golden - llm_unmatched_golden)),
        fp=len(llm_unmatched_golden - gt_unmatched_golden),
        fn=len(gt_unmatched_golden - llm_unmatched_golden),
    )

    asr_metrics = AlignmentMetrics(
        tp=len(gt_unmatched_asr & llm_unmatched_asr),
        tn=len((all_asr - gt_unmatched_asr) & (all_asr - llm_unmatched_asr)),
        fp=len(llm_unmatched_asr - gt_unmatched_asr),
        fn=len(gt_unmatched_asr - llm_unmatched_asr),
    )

    # Structural alignment accuracy
    structural_mismatches = compare_alignment_structures(gt_data, llm_data)
    total_golden = gt_data.get("total_golden_utterances", 0)
    structural_accuracy = (
        (total_golden - len(structural_mismatches)) / total_golden * 100 if total_golden else 0
    )

    duplicate_asr, duplicate_golden = check_duplicate_usage(llm_data)

    results = {
        "golden_utterances_metrics": golden_metrics.to_dict(),
        "asr_results_metrics": asr_metrics.to_dict(),
        "total_golden_utterances": total_golden,
        "total_asr_results": gt_data.get("total_asr_results", 0),
        "unmatched_golden_groundtruth": sorted(list(gt_unmatched_golden)),
        "unmatched_golden_llm": sorted(list(llm_unmatched_golden)),
        "unmatched_asr_groundtruth": sorted(list(gt_unmatched_asr)),
        "unmatched_asr_llm": sorted(list(llm_unmatched_asr)),
        "structural_alignment_metrics": {
            "mismatches": structural_mismatches,
            "accuracy_percent": round(structural_accuracy, 2),
        },
        "quality_issues": {
            "duplicate_asr_usage": duplicate_asr,
            "duplicate_golden_usage": duplicate_golden,
            "has_quality_issues": bool(duplicate_asr or duplicate_golden or structural_mismatches),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation completed. Results saved to: {output_path}")
    logger.info(f"  Golden precision/recall/f1: {golden_metrics.precision:.3f} / {golden_metrics.recall:.3f} / {golden_metrics.f1:.3f}")
    logger.info(f"  ASR precision/recall/f1: {asr_metrics.precision:.3f} / {asr_metrics.recall:.3f} / {asr_metrics.f1:.3f}")
    logger.info(f"  Structural accuracy: {structural_accuracy:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate alignment against groundtruth")
    parser.add_argument("--groundtruth", "-t", required=True, help="Path to groundtruth alignment JSON")
    parser.add_argument("--llm", "-l", required=True, help="Path to LLM alignment JSON")
    parser.add_argument("--output", "-o", required=True, help="Path to save evaluation output")
    args = parser.parse_args()

    evaluate_alignment(Path(args.groundtruth), Path(args.llm), Path(args.output))


if __name__ == "__main__":
    main()
