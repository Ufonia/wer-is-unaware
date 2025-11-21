import os
from typing import Optional, Tuple

import dspy


def init_openrouter(
    task_model: str,
    reflection_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 4000,
) -> Tuple[dspy.LM, Optional[dspy.LM]]:
    """Configure DSPy to use OpenRouter."""
    base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for OpenRouter provider")

    task_lm = dspy.LM(
        f"openrouter/{task_model}",
        api_base=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
    )
    dspy.settings.configure(lm=task_lm)

    reflection_lm = None
    if reflection_model:
        reflection_lm = dspy.LM(
            f"openrouter/{reflection_model}",
            api_base=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    return task_lm, reflection_lm
