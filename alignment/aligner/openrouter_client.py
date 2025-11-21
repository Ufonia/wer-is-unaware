import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    """Thin wrapper over OpenAI SDK configured for OpenRouter."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
