import dspy

class ClinicalImpactAssessment(dspy.Signature):
    """Assess the clinical impact of transcription errors in medical conversations.
    
    Compare the ground truth conversation with the transcription conversation and determine
    if errors would affect patient care. Focus on THREE distinct severity levels.
    """

    ground_truth_conversation = dspy.InputField()
    transcription_conversation = dspy.InputField()
    reasoning = dspy.OutputField(desc="Brief clinical justification for the assessment.")
    clinical_impact = dspy.OutputField(
        desc="""Clinical impact class (return ONLY the number):
        0 = No impact: cosmetic differences only (punctuation, capitalization, filler words)
        1 = Minimal impact: some information missing/changed but NOT critical to diagnosis or treatment decisions  
        2 = Significant impact: missing/incorrect information that COULD affect diagnosis, treatment, or patient safety
        Return ONLY: 0, 1, or 2"""
    )


class ClinicalImpactJudge(dspy.Module):
    """LLM Judge for assessing clinical impact."""

    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought(ClinicalImpactAssessment)

    def forward(self, ground_truth_conversation, transcription_conversation):
        return self.assess(
            ground_truth_conversation=ground_truth_conversation,
            transcription_conversation=transcription_conversation,
        )
