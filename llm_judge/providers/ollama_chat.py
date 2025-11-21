from typing import Optional, Tuple

import dspy


def init_ollama_chat(
    task_model: str,
    reflection_model: Optional[str] = None,
    max_tokens: int = 1000,
) -> Tuple[dspy.LM, Optional[dspy.LM]]:
    """Configure DSPy to use Ollama Chat."""
    task_lm = dspy.LM(f"ollama_chat/{task_model}", api_base='http://localhost:11434', api_key='', max_tokens=max_tokens)
    dspy.settings.configure(lm=task_lm)

    reflection_lm = None
    if reflection_model:
        reflection_lm = dspy.LM(
            f"ollama_chat/{reflection_model}", api_base='http://localhost:11434', api_key='', max_tokens=max_tokens
        )
    return task_lm, reflection_lm
