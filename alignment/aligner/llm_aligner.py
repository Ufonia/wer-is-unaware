"""
LLM-based transcript alignment using OpenRouter-powered models.
"""

import json
import logging
from typing import List

from .align_terms import AlignmentMatch, ASRResult, GoldenUtterance, TextAligner

logger = logging.getLogger(__name__)


class LLMTextAligner(TextAligner):
    """LLM-based aligner that aligns golden utterances with ASR results."""

    def __init__(
        self,
        llm_client,
        similarity_threshold: float = 0.3,
        fuzzy_threshold: float = 0.6,
        confidence_weight: float = 0.2,
    ):
        super().__init__(similarity_threshold, fuzzy_threshold, confidence_weight)
        self.llm_client = llm_client

    def align_utterances_with_llm(
        self, golden_utterances: List[GoldenUtterance], asr_results: List[ASRResult]
    ) -> List[AlignmentMatch]:
        """Use LLM to align golden utterances with ASR results."""
        if not golden_utterances or not asr_results:
            return []

        prompt = self._create_alignment_prompt(golden_utterances, asr_results)

        try:
            alignment_response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=65536,
            )

            alignments = self._parse_llm_response(alignment_response, golden_utterances, asr_results)
            alignments = self._fix_duplicate_asr_usage(alignments)
            alignments = self._catch_obvious_misses(alignments, golden_utterances, asr_results)
            return alignments

        except Exception as e:
            logger.error(f"Error calling LLM for alignment: {e}. Falling back to algorithmic alignment.")
            return super().align_utterances(golden_utterances, asr_results)

    def _create_alignment_prompt(self, golden_utterances: List[GoldenUtterance], asr_results: List[ASRResult]) -> str:
        """Create a structured prompt for LLM alignment."""
        golden_text = "GOLDEN TRANSCRIPT (Patient utterances):\n"
        for i, utterance in enumerate(golden_utterances):
            golden_text += f"G{i}: [{utterance.timestamp}] {utterance.text}\n"

        asr_text = "\nASR HYPOTHESIS (Patient utterances):\n"
        for i, result in enumerate(asr_results):
            asr_text += f"A{i}: {result.text} (confidence: {result.confidence:.2f})\n"

        prompt = f"""You are an expert at aligning speech transcripts. I need you to match patient utterances from a golden transcript with ASR (Automatic Speech Recognition) hypothesis results.

{golden_text}
{asr_text}

TASK: Align each golden transcript utterance (G0, G1, etc.) with the most appropriate ASR result(s) (A0, A1, etc.).

RULES:
1. You can only use utterances that exist in the input - DO NOT create new text
2. Each golden utterance should be matched to one ASR result, multiple ASR results, or marked as missing
3. Each ASR result can only be used ONCE - no ASR result should appear in multiple alignments
4. Make reasonable fuzzy matches even if the text isn't perfect - ASR often has errors
5. Consider semantic similarity, temporal proximity, and confidence scores
6. Multiple consecutive ASR results can be combined to match one golden utterance if they represent fragments
7. IMPORTANT: If an ASR result contains content that spans multiple consecutive golden utterances, those golden utterances should ALL be matched to that same ASR result

EXAMPLE of rule 7:
- Golden G5: "I know I understand that"
- Golden G6: "but it's different with the cataract"
- ASR A3: "I know I understand that but it's different with the cataract"
- CORRECT: G5→A3, G6→A3 (both use same ASR)
- WRONG: G5→missing, G6→A3 (creates artificial missing)

OUTPUT FORMAT (JSON):
{{
  "alignments": [
    {{
      "golden_index": 0,
      "asr_indices": [0],
      "match_type": "exact|fuzzy|missing",
      "similarity_score": 0.95,
      "explanation": "Brief reason for this alignment"
    }},
    ...
  ]
}}

Provide only the JSON response, no other text."""

        return prompt

    def _parse_llm_response(
        self, response: str, golden_utterances: List[GoldenUtterance], asr_results: List[ASRResult]
    ) -> List[AlignmentMatch]:
        """Parse LLM response into AlignmentMatch objects."""
        alignments = []

        try:
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response)
            alignment_data = data.get("alignments", [])

            for align_item in alignment_data:
                golden_idx = align_item["golden_index"]
                asr_indices = align_item.get("asr_indices", [])
                match_type = align_item.get("match_type", "fuzzy")
                similarity_score = align_item.get("similarity_score", 0.5)

                if golden_idx >= len(golden_utterances):
                    continue

                golden = golden_utterances[golden_idx]

                if not asr_indices:
                    alignments.append(
                        AlignmentMatch(
                            golden=golden,
                            asr=None,
                            asr_fragments=None,
                            similarity_score=0.0,
                            match_type="missing",
                        )
                    )
                elif len(asr_indices) == 1:
                    asr_idx = asr_indices[0]
                    if asr_idx < len(asr_results):
                        asr = asr_results[asr_idx]
                        alignments.append(
                            AlignmentMatch(
                                golden=golden,
                                asr=asr,
                                asr_fragments=None,
                                similarity_score=similarity_score,
                                match_type=match_type,
                            )
                        )
                else:
                    fragments = []
                    for asr_idx in asr_indices:
                        if asr_idx < len(asr_results):
                            fragments.append(asr_results[asr_idx])

                    if fragments:
                        combined_text = " ".join(f.text for f in fragments)
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
                                similarity_score=similarity_score,
                                match_type="multi_fragment",
                            )
                        )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            for golden in golden_utterances:
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
    
    def _fix_duplicate_asr_usage(self, alignments: List[AlignmentMatch]) -> List[AlignmentMatch]:
        """Fix cases where the same ASR result is used in multiple alignments"""
        from collections import defaultdict
        
        asr_usage = defaultdict(list)
        
        for i, alignment in enumerate(alignments):
            if alignment.asr and alignment.match_type != 'missing':
                # Use ASR ID as key, but fallback to sequence order for combined results
                asr_key = alignment.asr.id if alignment.asr.id != "combined" else f"seq_{alignment.asr.sequence_order}"
                asr_usage[asr_key].append((i, alignment))
        
        fixed_alignments = alignments.copy()
        
        for asr_key, alignment_list in asr_usage.items():
            if len(alignment_list) > 1:
                logger.warning(f"Duplicate ASR usage detected for {asr_key}: {len(alignment_list)} alignments")
                
                golden_texts = [align[1].golden.text for align in alignment_list]
                indices = [align[0] for align in alignment_list]
                
                logger.debug(f"Golden texts: {golden_texts}")
                
                # Check if these are consecutive golden utterances that should be combined
                consecutive_indices = sorted(indices)
                if all(consecutive_indices[i] == consecutive_indices[i-1] + 1 for i in range(1, len(consecutive_indices))):
                    combined_golden_text = " ".join(align[1].golden.text for align in sorted(alignment_list, key=lambda x: x[0]))
                    
                    first_idx = consecutive_indices[0]
                    first_alignment = fixed_alignments[first_idx]
                    
                    # Create a new GoldenUtterance with combined text but keep original timestamp
                    combined_golden = GoldenUtterance(
                        text=combined_golden_text,
                        timestamp=first_alignment.golden.timestamp,
                        sequence_order=first_alignment.golden.sequence_order
                    )
                    
                    fixed_alignments[first_idx] = AlignmentMatch(
                        golden=combined_golden,
                        asr=first_alignment.asr,
                        asr_fragments=first_alignment.asr_fragments,
                        similarity_score=first_alignment.similarity_score,
                        match_type=first_alignment.match_type
                    )
                    
                    logger.info(f"Combined consecutive golden texts into single alignment:")
                    logger.info(f"  Golden: '{combined_golden_text}'")
                    logger.info(f"  ASR: '{first_alignment.asr.text}'")
                    
                    combined_golden_indices = consecutive_indices.copy()
                    
                    fixed_alignments[first_idx].multi_golden_sequence = combined_golden_indices
                    fixed_alignments[first_idx].match_type = 'multi_golden'
                    
                    for idx in consecutive_indices[1:]:
                        fixed_alignments[idx] = None
                        logger.debug(f"Removed golden index {idx} (now part of combined match at index {first_idx})")
                        
                else:
                    # Not consecutive - keep the best match, mark others as missing
                    best_alignment_info = max(alignment_list, key=lambda x: x[1].similarity_score)
                    best_idx = best_alignment_info[0]
                    
                    logger.info(f"Keeping best match at golden index {best_idx} (score: {best_alignment_info[1].similarity_score:.3f})")
                    
                    for idx, alignment in alignment_list:
                        if idx != best_idx:
                            fixed_alignments[idx] = AlignmentMatch(
                                golden=alignment.golden,
                                asr=None,
                                asr_fragments=None,
                                similarity_score=0.0,
                                match_type='missing'
                            )
                            logger.debug(f"Marked golden index {idx} as missing")
        
        final_alignments = [alignment for alignment in fixed_alignments if alignment is not None]
        
        return final_alignments
    
    def _catch_obvious_misses(self, alignments: List[AlignmentMatch], 
                             golden_utterances: List[GoldenUtterance], 
                             asr_results: List[ASRResult]) -> List[AlignmentMatch]:
        """Simple post-processing to catch obvious missed matches between unmatched golden and unused ASR"""
        
        unmatched_indices = []
        for i, alignment in enumerate(alignments):
            if alignment.match_type == 'missing':
                unmatched_indices.append(i)
        
        if not unmatched_indices:
            return alignments
        
        used_asr_ids = set()
        for alignment in alignments:
            if alignment.asr and alignment.match_type != 'missing':
                used_asr_ids.add(alignment.asr.id)
        
        unused_asr = [asr for asr in asr_results if asr.id not in used_asr_ids]
        
        if not unused_asr:
            return alignments
        # TODO - check why the length of unused asr here doesnt appear to match the results in the unmatched analysis
        logger.info(f"Checking {len(unmatched_indices)} unmatched golden vs {len(unused_asr)} unused ASR for obvious matches...")
        
        fixed_alignments = alignments.copy()
        
        for golden_idx in unmatched_indices:
            golden = alignments[golden_idx].golden
            best_asr = None
            best_score = 0.0
            
            for asr in unused_asr:
                score = self.calculate_text_similarity(golden.text, asr.text)
                # TODO - consider making more strict - this does not take into account relative position of utterances, so it is matching utterances from completely different parts of the conversation
                # Perhaps, bring in the positional awareness here from the text alignment system
                if score > best_score and score >= 0.65:
                    best_score = score
                    best_asr = asr
            
            if best_asr:
                logger.info(f"Found obvious match:")
                logger.info(f"  Golden: '{golden.text}'")
                logger.info(f"  ASR: '{best_asr.text}' (similarity: {best_score:.3f})")
                
                fixed_alignments[golden_idx] = AlignmentMatch(
                    golden=golden,
                    asr=best_asr,
                    asr_fragments=None,
                    similarity_score=best_score,
                    match_type='fuzzy'
                )
                
                unused_asr.remove(best_asr)
        
        return fixed_alignments


def align_transcripts_with_llm(
    golden_file: str,
    asr_file: str,
    llm_client,
    similarity_threshold: float = 0.3,
    fuzzy_threshold: float = 0.6,
    confidence_weight: float = 0.2,
) -> List[AlignmentMatch]:
    """Align golden transcript with ASR results using the LLM aligner."""
    with open(golden_file, "r") as f:
        golden_data = json.load(f)

    with open(asr_file, "r") as f:
        asr_data = json.load(f)

    aligner = LLMTextAligner(llm_client, similarity_threshold, fuzzy_threshold, confidence_weight)
    golden_utterances = aligner.parse_golden_transcript(golden_data)
    asr_results = aligner.parse_asr_results(asr_data)

    logger.info(f"Loaded {len(golden_utterances)} golden utterances and {len(asr_results)} ASR results")
    return aligner.align_utterances_with_llm(golden_utterances, asr_results)
