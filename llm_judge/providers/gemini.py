import os
from typing import Optional, Tuple

import dspy
import vertexai


def init_gemini(
    task_model: str,
    reflection_model: Optional[str] = None,
    max_tokens: int = 8000,
) -> Tuple[dspy.LM, Optional[dspy.LM]]:
    """Configure DSPy to use Gemini via Vertex AI."""
    project = os.getenv("GCP_PROJECT_ID", "your-project-id")
    location = os.getenv("GCP_LOCATION", "us-central1")
    vertexai.init(project=project, location=location)

    task_lm = dspy.LM(f"vertex_ai/{task_model}", max_tokens=max_tokens)
    dspy.settings.configure(lm=task_lm)

    reflection_lm = None
    if reflection_model:
        reflection_lm = dspy.LM(f"vertex_ai/{reflection_model}", max_tokens=max_tokens)
    return task_lm, reflection_lm
