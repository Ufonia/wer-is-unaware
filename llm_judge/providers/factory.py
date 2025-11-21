from typing import Optional

from .bedrock import init_bedrock
from .gemini import init_gemini
from .openrouter import init_openrouter
from .ollama_chat import init_ollama_chat


def setup_models(
    provider: str,
    task_model: str,
    reflection_model: Optional[str] = None,
    **kwargs,
):
    """Initialize task/reflection LMs and configure DSPy."""
    provider = provider.lower()
    if provider == "gemini":
        return init_gemini(task_model, reflection_model=reflection_model, **kwargs)
    if provider == "bedrock":
        return init_bedrock(task_model, reflection_model=reflection_model, **kwargs)
    if provider == "openrouter":
        return init_openrouter(task_model, reflection_model=reflection_model, **kwargs)
    if provider == "ollam_chat":
        return init_ollama_chat(task_model, reflection_model=reflection_model, **kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
