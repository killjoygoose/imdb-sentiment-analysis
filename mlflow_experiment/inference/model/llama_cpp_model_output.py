from dataclasses import dataclass


@dataclass
class LlamaCppModelOutput:
    content: str
    prompt_tokens: int
    completion_tokens: int
