"""
This module holds an abstract class to unify all wrappers around llama.cpp.
It could be the case that we want to e.g. to query llama multiple times
for the so-called "self-consistency" prompting (majority vote) or
could query a database to get
"""

from abc import ABC, abstractmethod
from typing import Any

from llama_cpp import Llama

from mlflow_experiment.inference.model.llama_cpp_model_output import LlamaCppModelOutput
from mlflow_experiment.inference.prompt_building import UserPromptBuilder


class LlamaCppModel(ABC):
    def __init__(
        self, model: Llama, user_prompt_builder: UserPromptBuilder, model_name: str
    ):
        self.model = model
        self.user_prompt_builder = user_prompt_builder
        self.model_name = model_name

    @property
    def base_user_prompt(self) -> str:
        return self.user_prompt_builder.base_user_prompt

    @abstractmethod
    def run(
        self,
        system_prompt: str,
        user_prompt_content: dict[str, Any],
        **inference_kwargs
    ) -> LlamaCppModelOutput:
        """
        Perform inference with llama.cpp over the messages set.
        """
