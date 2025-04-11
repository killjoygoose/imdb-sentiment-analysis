from typing import Any
from mlflow_experiment.inference.prompt_building.user_prompt_builder import (
    UserPromptBuilder,
)


class BasicUserPromptBuilder(UserPromptBuilder):
    """
    A basic user prompt builder that just uses string formatting
    to insert the user prompt content at the respective fields.
    """

    def _build_user_prompt(self, user_prompt_content: dict[str, Any] = None) -> str:
        return self.base_user_prompt.format(**user_prompt_content or {})
