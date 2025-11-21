import json
import re
from typing import Optional

import dspy

COST_MATRIX = [
    [1.2, 0.3, -1.0],
    [0.3, 1.5, 0.5],
    [-1.2, 0.4, 1.5],
]

def parse_label(label_str: str) -> Optional[int]:
    try:
        label_str = str(label_str).strip()
        if label_str in {"0", "1", "2"}:
            return int(label_str)

        json_match = re.search(r"\{.*\}", label_str, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(0))
            val = obj.get("clinical_impact")
            if val in [0, 1, 2] or str(val) in "012":
                return int(val)

        num_match = re.search(r"\b([0-2])\b", label_str)
        if num_match:
            return int(num_match.group(1))
    except Exception:
        return None
    return None

def gepa_feedback_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    true_label = int(example.clinical_impact)
    pred_label = parse_label(prediction.clinical_impact)
    
    if pred_label is None:
        feedback = (
            f"PARSING ERROR: The model failed to output a valid class (0, 1, or 2). "
            f"Raw output: '{prediction.clinical_impact}'. "
            f"The model MUST return ONLY the number 0, 1, or 2 as specified in the output field description. "
            f"Consider emphasizing in the instructions: output format must be strictly a single digit."
        )
        return dspy.Prediction(score=-2.0, feedback=feedback)
    
    score = COST_MATRIX[true_label][pred_label]
    
    # Generate detailed feedback based on the prediction outcome
    if pred_label == true_label:
        class_names = {0: "No impact", 1: "Minimal impact", 2: "Significant impact"}
        feedback = (
            f"CORRECT: Correctly identified as Class {true_label} ({class_names[true_label]}). "
            f"The model's reasoning was appropriate for this classification. "
            f"Continue using similar reasoning patterns for this type of case."
        )
    else:
        if true_label == 0 and pred_label > 0:
            feedback = (
                f"OVER-CLASSIFICATION: Predicted Class {pred_label} but should be Class 0 (No impact). "
                f"The transcription differences are cosmetic (punctuation, capitalization, filler words) "
                f"and do NOT affect clinical meaning. The model should be MORE LENIENT with minor differences "
                f"and focus ONLY on content that affects diagnosis or treatment decisions."
            )
        elif true_label == 1 and pred_label == 0:
            feedback = (
                f"UNDER-CLASSIFICATION: Predicted Class 0 but should be Class 1 (Minimal impact). "
                f"While not critical to diagnosis/treatment, some clinically relevant information was "
                f"missing or changed. The model should be MORE SENSITIVE to information changes, "
                f"even if they don't directly affect critical decisions."
            )
        elif true_label == 1 and pred_label == 2:
            feedback = (
                f"OVER-CLASSIFICATION: Predicted Class 2 but should be Class 1 (Minimal impact). "
                f"The information changes are not critical enough to affect diagnosis or patient safety. "
                f"Reserve Class 2 ONLY for errors that COULD directly affect diagnosis, treatment, or safety. "
                f"The model should distinguish between 'some information missing' vs 'critical information missing'."
            )
        elif true_label == 2 and pred_label < 2:
            feedback = (
                f"CRITICAL MISS: Predicted Class {pred_label} but should be Class 2 (Significant impact). "
                f"This is a HIGH-PRIORITY error. The transcription contained missing/incorrect information "
                f"that COULD affect diagnosis, treatment, or patient safety. The model MUST be MORE SENSITIVE "
                f"to clinically critical information like symptoms, medications, measurements, or diagnoses. "
                f"Look for: changed medical terms, missing symptoms, altered measurements, or omitted diagnoses."
            )
        else:
            feedback = (
                f"MAJOR ERROR: Predicted Class {pred_label} but should be Class {true_label}. "
                f"This is a large classification error spanning 2 severity levels. "
                f"The model needs to fundamentally reassess its criteria for clinical impact. "
                f"Review the distinction between cosmetic changes, information changes, and critical errors."
            )
    
    feedback += f" [True: {true_label}, Predicted: {pred_label}]"
    
    return dspy.Prediction(score=score, feedback=feedback)


def simple_metric(example, prediction, trace=None):
    true_label = int(example.clinical_impact)
    pred_label = parse_label(prediction.clinical_impact)
    if pred_label is None:
        return -2.0
    return COST_MATRIX[true_label][pred_label]
