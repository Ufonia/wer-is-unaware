from typing import Optional, Tuple

import dspy

def init_bedrock(
    task_model: str,
    reflection_model: Optional[str] = None,
    region: str = "us-east-1",
    max_tokens: int = 1000,
) -> Tuple[dspy.LM, Optional[dspy.LM]]:
    """Configure DSPy to use AWS Bedrock."""
    task_lm = dspy.LM(f"bedrock/{task_model}", region_name=region, max_tokens=max_tokens)
    dspy.settings.configure(lm=task_lm)

    reflection_lm = None
    if reflection_model:
        reflection_lm = dspy.LM(
            f"bedrock/{reflection_model}", region_name=region, max_tokens=max_tokens
        )
    return task_lm, reflection_lm
