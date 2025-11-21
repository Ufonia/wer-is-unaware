import json
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

from num2words import num2words


def _convert_numbers_to_words(text: str) -> str:
    """Convert numeric ordinals/cardinals to words (en_GB)."""
    lang = "en_GB"
    text = re.sub(
        r"(\d+)(st|nd|rd|th)",
        lambda m: num2words(int(m.group(1)), to="ordinal", lang=lang),
        text,
    )
    text = re.sub(
        r"(\d+(\.\d+)?)",
        lambda m: num2words(m.group(1), lang=lang),
        text,
    )
    return text


@dataclass
class GoldenUtterance:
    """Represents a patient utterance from the golden transcript."""

    text: str
    timestamp: str
    sequence_order: int


@dataclass
class ASRResult:
    """Represents an ASR hypothesis result."""

    id: str
    text: str
    confidence: float
    started_at: str
    ended_at: str
    received_at: str
    sequence_order: int


@dataclass
class AlignmentMatch:
    """Represents an aligned pair of golden utterance and ASR result(s)."""

    golden: GoldenUtterance
    asr: Optional[ASRResult]
    asr_fragments: Optional[List[ASRResult]]  # For multi-fragment matches
    similarity_score: float
    match_type: str  # 'exact', 'fuzzy', 'missing', 'multi_fragment'


class TextAligner:
    """Aligns golden transcript patient utterances with ASR hypothesis results."""

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        fuzzy_threshold: float = 0.6,
        confidence_weight: float = 0.2,
    ):
        self.similarity_threshold = similarity_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.confidence_weight = confidence_weight

    def parse_golden_transcript(
        self, golden_data: Dict, key: str = "transcript_golden_anonymized"
    ) -> List[GoldenUtterance]:
        """Extract patient utterances from golden transcript in sequential order."""
        transcript = golden_data.get(key, "")

        patient_pattern = r"\[(\d{2}:\d{2})\] Patient: (.+?)(?=\n\[|\n$|$)"
        matches = re.findall(patient_pattern, transcript, re.DOTALL)

        utterances = []
        for i, (timestamp, text) in enumerate(matches):
            cleaned_text = re.sub(r"\s+", " ", text.strip())
            utterances.append(
                GoldenUtterance(text=cleaned_text, timestamp=timestamp, sequence_order=i)
            )

        return utterances

    def parse_asr_results(self, asr_data: List[Dict]) -> List[ASRResult]:
        """Parse and sort ASR results by chronological order."""
        results = []
        for i, result in enumerate(asr_data):
            results.append(
                ASRResult(
                    id=result["id"],
                    text=result["text"].strip(),
                    confidence=result["confidence"],
                    started_at=result["startedAt"],
                    ended_at=result["endedAt"],
                    received_at=result["receivedAt"],
                    sequence_order=i,
                )
            )

        def sort_key(x):
            if x.started_at is not None:
                return datetime.fromisoformat(x.started_at.replace("Z", "+00:00"))
            return datetime.fromisoformat(x.received_at.replace("Z", "+00:00"))

        results.sort(key=sort_key)

        for i, result in enumerate(results):
            result.sequence_order = i

        return results

    def _resolve_proxy_key(self, golden_data: Dict) -> str:
        """
        Resolve the proxy transcript key with backward compatibility.

        Priority:
        1. transcript_proxy_anonymized (new standard)
        2. transcript_hypothesis_anonymized (human annotation)
        3. transcript_gemini_anonymized (old format)
        """
        proxy_keys = [
            "transcript_proxy_anonymized",
            "transcript_hypothesis_anonymized",
            "transcript_gemini_anonymized",
        ]

        for key in proxy_keys:
            if key in golden_data:
                return key
        return "transcript_proxy_anonymized"

    def convert_golden_to_asr_format_for_proxy(
        self, golden_utterances: List[GoldenUtterance]
    ) -> List[ASRResult]:
        """Convert golden utterances to ASR format for proxy alignment."""
        asr_results = []
        for i, utterance in enumerate(golden_utterances):
            asr_result = ASRResult(
                id=f"proxy_{i}",
                text=utterance.text,
                confidence=1.0,
                started_at=utterance.timestamp,
                ended_at=utterance.timestamp,
                received_at=utterance.timestamp,
                sequence_order=i,
            )
            asr_results.append(asr_result)
        return asr_results

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods."""
        if not text1 or not text2:
            return 0.0

        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)

        char_similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()

        words1 = set(text1_norm.lower().split())
        words2 = set(text2_norm.lower().split())
        if words1 or words2:
            word_similarity = len(words1 & words2) / len(words1 | words2)
        else:
            word_similarity = 0.0

        len1, len2 = len(text1_norm), len(text2_norm)
        if max(len1, len2) > 0:
            length_similarity = 1 - abs(len1 - len2) / max(len1, len2)
        else:
            length_similarity = 1.0

        combined_similarity = (
            0.5 * char_similarity + 0.3 * word_similarity + 0.2 * length_similarity
        )

        if len(words1) <= 3 or len(words2) <= 3:
            if word_similarity == 0.0:
                common_words = {"or", "and", "the", "a", "an", "is", "are", "was", "were", "yes", "no"}
                meaningful_words1 = words1 - common_words
                meaningful_words2 = words2 - common_words
                if len(meaningful_words1) > 0 and len(meaningful_words2) > 0:
                    return 0.0

        return combined_similarity

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text.strip())
        text = text.replace(" and ", " & ")
        text = text.replace(" or ", " | ")
        text = _convert_numbers_to_words(text)
        return text

    def calculate_match_score(
        self,
        golden: GoldenUtterance,
        asr: ASRResult,
        golden_idx: int = None,
        asr_idx: int = None,
    ) -> float:
        """Calculate overall match score between golden and ASR with proximity weighting."""
        text_similarity = self.calculate_text_similarity(golden.text, asr.text)
        confidence_boost = asr.confidence * self.confidence_weight
        base_score = min(1.0, text_similarity + confidence_boost)

        if golden_idx is not None and asr_idx is not None:
            distance = abs(golden_idx - asr_idx)
            if distance <= 2:
                proximity_penalty = 0.0
            elif distance <= 5:
                proximity_penalty = (distance - 2) * 0.1
            else:
                proximity_penalty = min(0.4, 0.3 + (distance - 5) * 0.02)
            return base_score * (1 - proximity_penalty)
        return base_score

    def calculate_multi_fragment_score(
        self, golden: GoldenUtterance, asr_fragments: List[ASRResult], golden_idx: int = None
    ) -> float:
        """Calculate match score between golden utterance and multiple ASR fragments."""
        if not asr_fragments:
            return 0.0

        combined_text = " ".join(fragment.text for fragment in asr_fragments)
        text_similarity = self.calculate_text_similarity(golden.text, combined_text)
        avg_confidence = sum(fragment.confidence for fragment in asr_fragments) / len(asr_fragments)
        confidence_boost = avg_confidence * self.confidence_weight
        base_score = min(1.0, text_similarity + confidence_boost)

        if golden_idx is not None:
            first_asr_idx = asr_fragments[0].sequence_order
            distance = abs(golden_idx - first_asr_idx)
            if distance <= 2:
                proximity_penalty = 0.0
            elif distance <= 5:
                proximity_penalty = (distance - 2) * 0.1
            else:
                proximity_penalty = min(0.4, 0.3 + (distance - 5) * 0.02)
            return base_score * (1 - proximity_penalty)
        return base_score

    def align_utterances(
        self, golden_utterances: List[GoldenUtterance], asr_results: List[ASRResult]
    ) -> List[AlignmentMatch]:
        """Align golden utterances with ASR results using sequential matching."""
        anchors = self._find_anchor_matches(golden_utterances, asr_results)
        alignments = self._fill_alignment_gaps(golden_utterances, asr_results, anchors)
        return alignments

    def _find_anchor_matches(
        self, golden_utterances: List[GoldenUtterance], asr_results: List[ASRResult]
    ) -> List[Tuple[int, int, float]]:
        """Find high-confidence matches to use as alignment anchors."""
        anchors = []
        for g_idx, golden in enumerate(golden_utterances):
            for a_idx, asr in enumerate(asr_results):
                score = self.calculate_match_score(golden, asr, g_idx, a_idx)
                if score > 0.95 and asr.confidence > 0.95:
                    anchors.append((g_idx, a_idx, score))

        anchors.sort(key=lambda x: x[2], reverse=True)

        final_anchors = []
        used_golden = set()
        used_asr = set()
        for g_idx, a_idx, score in anchors:
            if g_idx in used_golden or a_idx in used_asr:
                continue

            golden_text = golden_utterances[g_idx].text.lower().strip()
            asr_text = asr_results[a_idx].text.lower().strip()
            if self._is_valid_anchor_match(golden_text, asr_text, score):
                final_anchors.append((g_idx, a_idx, score))
                used_golden.add(g_idx)
                used_asr.add(a_idx)

        final_anchors.sort(key=lambda x: x[0])
        return final_anchors

    def _is_valid_anchor_match(self, golden_text: str, asr_text: str, score: float) -> bool:
        """Validate that an anchor match makes semantic sense."""
        golden_words = set(golden_text.split())
        asr_words = set(asr_text.split())

        if len(golden_words) > 0 and len(asr_words) > 0:
            overlap = len(golden_words & asr_words)
            union = len(golden_words | asr_words)
            word_similarity = overlap / union if union > 0 else 0.0
            if score > 0.95 and word_similarity < 0.3:
                actual_sim = self.calculate_text_similarity(golden_text, asr_text)
                if actual_sim < 0.7:
                    return False
        return True

    def _fill_alignment_gaps(
        self,
        golden_utterances: List[GoldenUtterance],
        asr_results: List[ASRResult],
        anchors: List[Tuple[int, int, float]],
    ) -> List[AlignmentMatch]:
        """Fill in alignments between anchor points."""
        alignments = []
        used_asr_indices = set()

        anchor_dict = {}
        for g_idx, a_idx, score in anchors:
            anchor_dict[g_idx] = (a_idx, score)
            used_asr_indices.add(a_idx)

        for g_idx, golden in enumerate(golden_utterances):
            if g_idx in anchor_dict:
                a_idx, score = anchor_dict[g_idx]
                alignments.append(
                    AlignmentMatch(
                        golden=golden,
                        asr=asr_results[a_idx],
                        asr_fragments=None,
                        similarity_score=score,
                        match_type="exact",
                    )
                )
                continue

            best_match = self._find_best_match_in_range(
                golden, asr_results, g_idx, anchors, used_asr_indices
            )

            if best_match:
                match_result, score, match_type_hint = best_match
                if match_type_hint == "multi_fragment":
                    fragments = match_result
                    if score >= self.fuzzy_threshold:
                        for fragment in fragments:
                            used_asr_indices.add(fragment.sequence_order)

                        combined_text = " ".join(fragment.text for fragment in fragments)
                        primary_asr = ASRResult(
                            id="combined",
                            text=combined_text,
                            confidence=sum(f.confidence for f in fragments) / len(fragments),
                            started_at=fragments[0].started_at,
                            ended_at=fragments[-1].ended_at,
                            received_at=fragments[0].received_at,
                            sequence_order=fragments[0].sequence_order,
                        )

                        alignments.append(
                            AlignmentMatch(
                                golden=golden,
                                asr=primary_asr,
                                asr_fragments=fragments,
                                similarity_score=score,
                                match_type="multi_fragment",
                            )
                        )
                    else:
                        alignments.append(
                            AlignmentMatch(
                                golden=golden,
                                asr=None,
                                asr_fragments=None,
                                similarity_score=0.0,
                                match_type="missing",
                            )
                        )
                else:
                    asr = match_result
                    if score >= self.fuzzy_threshold:
                        used_asr_indices.add(asr.sequence_order)
                        alignments.append(
                            AlignmentMatch(
                                golden=golden,
                                asr=asr,
                                asr_fragments=None,
                                similarity_score=score,
                                match_type="exact" if score > 0.8 else "fuzzy",
                            )
                        )
                    else:
                        alignments.append(
                            AlignmentMatch(
                                golden=golden,
                                asr=None,
                                asr_fragments=None,
                                similarity_score=0.0,
                                match_type="missing",
                            )
                        )
            else:
                alignments.append(
                    AlignmentMatch(
                        golden=golden,
                        asr=None,
                        asr_fragments=None,
                        similarity_score=0.0,
                        match_type="missing",
                    )
                )

        return alignments

    def _find_best_match_in_range(
        self,
        golden: GoldenUtterance,
        asr_results: List[ASRResult],
        golden_idx: int,
        anchors: List[Tuple[int, int, float]],
        used_asr_indices: set,
    ) -> Optional[Tuple[Union[ASRResult, List[ASRResult]], float, str]]:
        """Find best ASR match (single or multi-fragment) within the allowed range."""
        min_asr_idx = 0
        max_asr_idx = len(asr_results) - 1

        for g_idx, a_idx, _ in anchors:
            if g_idx < golden_idx:
                min_asr_idx = max(min_asr_idx, a_idx + 1)
            elif g_idx > golden_idx:
                max_asr_idx = min(max_asr_idx, a_idx - 1)
                break

        if min_asr_idx > max_asr_idx:
            center = (min_asr_idx + max_asr_idx) // 2
            window = max(5, len(asr_results) // 10)
            min_asr_idx = max(0, center - window)
            max_asr_idx = min(len(asr_results) - 1, center + window)

        available_asr = [
            asr
            for asr in asr_results
            if min_asr_idx <= asr.sequence_order <= max_asr_idx
            and asr.sequence_order not in used_asr_indices
        ]
        if not available_asr:
            return None

        best_match = None
        best_score = 0.0
        best_match_type = "single"

        for start_idx in range(len(available_asr)):
            for fragment_count in range(2, min(4, len(available_asr) - start_idx + 1)):
                fragments = available_asr[start_idx : start_idx + fragment_count]
                if self._are_consecutive_fragments(fragments):
                    score = self.calculate_multi_fragment_score(golden, fragments, golden_idx)
                    if score > best_score and score >= self.similarity_threshold:
                        best_match = fragments
                        best_score = score
                        best_match_type = "multi_fragment"

        for asr in available_asr:
            score = self.calculate_match_score(golden, asr, golden_idx, asr.sequence_order)
            if score > best_score and score >= self.similarity_threshold:
                best_match = asr
                best_score = score
                best_match_type = "single"

        if best_match:
            return (best_match, best_score, best_match_type)
        return None

    def _are_consecutive_fragments(self, fragments: List[ASRResult]) -> bool:
        """Check if ASR fragments are consecutive in sequence order."""
        if len(fragments) <= 1:
            return True
        for i in range(1, len(fragments)):
            if fragments[i].sequence_order != fragments[i - 1].sequence_order + 1:
                return False
        return True

    def print_alignment_results(
        self, alignments: List[AlignmentMatch], asr_results: List[ASRResult]
    ):
        """Print comprehensive alignment results with unmatched tracking."""
        print("=" * 80)
        print("ALIGNMENT RESULTS")
        print("=" * 80)

        exact_matches = sum(1 for a in alignments if a.match_type == "exact")
        fuzzy_matches = sum(1 for a in alignments if a.match_type == "fuzzy")
        missing_matches = sum(1 for a in alignments if a.match_type == "missing")

        used_asr_ids = set()
        for alignment in alignments:
            if alignment.asr:
                if alignment.match_type == "multi_fragment" and alignment.asr_fragments:
                    for fragment in alignment.asr_fragments:
                        used_asr_ids.add(fragment.id)
                else:
                    used_asr_ids.add(alignment.asr.id)

        unused_asr_results = [asr for asr in asr_results if asr.id not in used_asr_ids]

        print("Summary:")
        print(f"  Exact matches: {exact_matches}")
        print(f"  Fuzzy matches: {fuzzy_matches}")
        print(f"  Missing ASR: {missing_matches}")
        print(f"  Total golden utterances: {len(alignments)}")
        print(f"  Golden coverage: {((exact_matches + fuzzy_matches) / len(alignments) * 100):.1f}%")
        print(f"  Unused ASR results: {len(unused_asr_results)}")
        print(f"  ASR coverage: {(len(used_asr_ids) / len(asr_results) * 100):.1f}%")
        print()

        for i, alignment in enumerate(alignments):
            print(f"#{i+1} [{alignment.match_type.upper()}] Score: {alignment.similarity_score:.3f}")
            print(f"  Golden: '{alignment.golden.text}'")
            if alignment.asr:
                if alignment.match_type == "multi_fragment" and alignment.asr_fragments:
                    print(f"  ASR:    '{alignment.asr.text}' (avg conf: {alignment.asr.confidence:.3f})")
                    print("  Fragments:")
                    for j, fragment in enumerate(alignment.asr_fragments):
                        print(f"    [{j+1}] '{fragment.text}' (conf: {fragment.confidence:.3f})")
                else:
                    print(f"  ASR:    '{alignment.asr.text}' (conf: {alignment.asr.confidence:.3f})")
            else:
                print("  ASR:    [MISSING]")
            print()

        self._print_unmatched_analysis(alignments, unused_asr_results)

    def _print_unmatched_analysis(
        self, alignments: List[AlignmentMatch], unused_asr_results: List[ASRResult]
    ):
        """Print detailed analysis of unmatched items from both sides."""
        print("=" * 80)
        print("UNMATCHED ANALYSIS")
        print("=" * 80)

        unmatched_golden = [a for a in alignments if a.match_type == "missing"]

        if unmatched_golden:
            print(f"UNMATCHED GOLDEN UTTERANCES ({len(unmatched_golden)}):")
            for alignment in unmatched_golden:
                print(
                    f"  #{alignment.golden.sequence_order + 1} "
                    f"[{alignment.golden.timestamp}] '{alignment.golden.text}'"
                )
            print()
        else:
            print("All golden utterances have ASR matches!\n")

        if unused_asr_results:
            print(f"UNUSED ASR RESULTS ({len(unused_asr_results)}):")
            for asr in unused_asr_results:
                try:
                    start_time = datetime.fromisoformat(asr.started_at.replace("Z", "+00:00"))
                    time_str = start_time.strftime("%H:%M:%S")
                except Exception:
                    time_str = "unknown"
                print(f"  #{asr.sequence_order + 1} [{time_str}] '{asr.text}' (conf: {asr.confidence:.3f})")
            print()
        else:
            print("All ASR results are matched to golden utterances!\n")

        total_golden = len(alignments)
        total_asr = len(unused_asr_results) + sum(1 for a in alignments if a.asr is not None)
        matched_pairs = sum(1 for a in alignments if a.asr is not None)

        print("ANALYSIS SUMMARY:")
        print(f"  • Total Golden utterances: {total_golden}")
        print(f"  • Total ASR results: {total_asr}")
        print(f"  • Successfully matched pairs: {matched_pairs}")
        print(f"  • Golden lines missing ASR: {len(unmatched_golden)}")
        print(f"  • ASR lines without golden match: {len(unused_asr_results)}")

        if total_golden > 0:
            recall = (matched_pairs / total_golden) * 100
            print(f"  • Recall (Golden covered): {recall:.1f}%")

        if total_asr > 0:
            precision = (matched_pairs / total_asr) * 100
            print(f"  • Precision (ASR used): {precision:.1f}%")

        print("=" * 80)

    def export_unmatched_analysis(
        self,
        alignments: List[AlignmentMatch],
        asr_results: List[ASRResult],
        output_file: str = "data/unmatched_analysis.json",
    ):
        """Export detailed unmatched analysis to JSON file."""
        used_asr_ids = set()
        for alignment in alignments:
            if alignment.asr:
                if alignment.match_type == "multi_fragment" and alignment.asr_fragments:
                    for fragment in alignment.asr_fragments:
                        used_asr_ids.add(fragment.id)
                else:
                    used_asr_ids.add(alignment.asr.id)

        unused_asr_results = [asr for asr in asr_results if asr.id not in used_asr_ids]
        unmatched_golden = [a for a in alignments if a.match_type == "missing"]

        analysis_data = {
            "summary": {
                "total_golden_utterances": len(alignments),
                "total_asr_results": len(asr_results),
                "matched_pairs": sum(1 for a in alignments if a.asr is not None),
                "unmatched_golden_count": len(unmatched_golden),
                "unused_asr_count": len(unused_asr_results),
                "golden_coverage_percent": (
                    (len(alignments) - len(unmatched_golden)) / len(alignments) * 100
                )
                if alignments
                else 0,
                "asr_coverage_percent": (len(used_asr_ids) / len(asr_results) * 100)
                if asr_results
                else 0,
            },
            "unmatched_golden_utterances": [
                {
                    "sequence_order": alignment.golden.sequence_order,
                    "timestamp": alignment.golden.timestamp,
                    "text": alignment.golden.text,
                    "reason": "missing_from_asr",
                }
                for alignment in unmatched_golden
            ],
            "unused_asr_results": [
                {
                    "sequence_order": asr.sequence_order,
                    "id": asr.id,
                    "text": asr.text,
                    "confidence": asr.confidence,
                    "started_at": asr.started_at,
                    "ended_at": asr.ended_at,
                    "reason": "no_golden_match",
                }
                for asr in unused_asr_results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(analysis_data, f, indent=2)

        print(f"Unmatched analysis exported to: {output_file}")
        return analysis_data

    def save_alignment_results(self, alignments: List[AlignmentMatch], output_file_path: str):
        """Save alignment results to file."""
        with open(output_file_path, "w") as f:
            alignment_data = []
            for a in alignments:
                result = {
                    "golden_text_index": a.golden.sequence_order,
                    "golden_text": a.golden.text,
                    "golden_timestamp": a.golden.timestamp,
                    "asr_text_index": a.asr.sequence_order if a.asr else None,
                    "asr_text": a.asr.text if a.asr else None,
                    "asr_confidence": a.asr.confidence if a.asr else None,
                    "asr_fragments": None,
                    "similarity_score": a.similarity_score,
                    "match_type": a.match_type,
                }

                if hasattr(a, "multi_golden_sequence") and a.multi_golden_sequence:
                    result["multi_golden_sequence"] = a.multi_golden_sequence

                if a.asr_fragments:
                    result["asr_fragments"] = [
                        {
                            "index": f.sequence_order,
                            "text": f.text,
                            "confidence": f.confidence,
                            "started_at": f.started_at,
                            "ended_at": f.ended_at,
                        }
                        for f in a.asr_fragments
                    ]
                    result["asr_text_index"] = a.asr_fragments[0].sequence_order

                alignment_data.append(result)

            json.dump(alignment_data, f, indent=2)


def align_transcripts(
    golden_file: str,
    asr_file: str,
    similarity_threshold: float = 0.3,
    fuzzy_threshold: float = 0.6,
    confidence_weight: float = 0.2,
) -> List[AlignmentMatch]:
    """
    Align golden transcript with ASR results using the algorithmic aligner.
    """
    with open(golden_file, "r") as f:
        golden_data = json.load(f)

    with open(asr_file, "r") as f:
        asr_data = json.load(f)

    aligner = TextAligner(similarity_threshold, fuzzy_threshold, confidence_weight)
    golden_utterances = aligner.parse_golden_transcript(golden_data)
    asr_results = aligner.parse_asr_results(asr_data)

    alignments = aligner.align_utterances(golden_utterances, asr_results)
    aligner.print_alignment_results(alignments, asr_results)
    aligner.export_unmatched_analysis(alignments, asr_results, "data/unmatched_analysis.json")
    return alignments


if __name__ == "__main__":
    golden_file = "data/asr-transcripts/golden_sample.json"
    asr_file = "data/asr-transcripts/patient_sample.json"
    align_transcripts(golden_file, asr_file, fuzzy_threshold=0.6)
