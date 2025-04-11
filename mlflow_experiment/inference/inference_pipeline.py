"""
This module defines the `InferencePipeline` class, which handles generating
prompts, running inference using a llm, and applying
postprocessing to produce structured classification outputs. It supports
batch and single-prompt inference and logs timing and token usage.
"""

from typing import Any, Callable, Iterator, Iterable
import logging
from llama_cpp import Llama
from tqdm import tqdm

from mlflow_experiment.inference.postprocessing.binary_classification_output import (
    BinaryClassificationOutput,
)
from mlflow_experiment.inference.prompt_building import UserPromptBuilder
from mlflow_experiment.inference.inference_output import InferenceOutput
from time import time

_LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    """
    Class which covers the whole inference process for the current binary
    classification task.

    Args:
        model (Llama): The model used for inference.
        system_prompt (str): The system prompt.
        user_prompt_builder (UserPromptBuilder): The user prompt builder.
        postprocessing_fn (optional, Callable[[str], str]): The function that parses the output
            of the LLM and returns the relevant string from it. It should return a string, as we
            should keep the option to evaluate the number of hallucinated or incorrectly
            formatted outputs.
    """

    def __init__(
        self,
        model: Llama,
        system_prompt: str,
        user_prompt_builder: UserPromptBuilder,
        postprocessing_fn: Callable[[str], BinaryClassificationOutput],
    ):
        self.model = model
        self._system_prompt = system_prompt
        self.user_prompt_builder = user_prompt_builder
        self.postprocessing_fn = postprocessing_fn

    @property
    def base_user_prompt(self) -> str:
        return self.user_prompt_builder.base_user_prompt

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def run_single(
        self, user_prompt_content: dict[str, Any], **inference_kwargs
    ) -> InferenceOutput:
        """
        Run inference on a single data point.
        """
        user_prompt = self.user_prompt_builder.build_user_prompt(user_prompt_content)
        _start = time()

        output = self.model.create_chat_completion(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **inference_kwargs
        )
        output_string = output["choices"][0]["message"]["content"]

        classification_output = self.postprocessing_fn(output_string)

        return InferenceOutput(
            output_string=output_string,
            input_token_count=output["usage"]["prompt_tokens"],
            output_token_count=output["usage"]["completion_tokens"],
            elapsed_time=time() - _start,
            is_hallucination=classification_output.is_hallucination,
            is_wrong_format=classification_output.is_wrong_format,
            label=classification_output.label,
        )

    def run(
        self, user_prompt_contents: Iterable[dict[str, Any]], **inference_kwargs
    ) -> Iterator[InferenceOutput]:
        """
        Bulk inference.

        Args:
            user_prompt_contents (Iterable[dict[str, Any]]): list of prompt contents.
            **inference_kwargs: Additional parameters for the model.

        Returns:
            Iterator[InferenceOutput] the goal is to be able to inspect output by output
                without having necessarily to first wait for the whole process to end
                on the whole dataset.
        """
        user_prompt_contents = (
            user_prompt_contents if user_prompt_contents is not None else [{}]
        )

        for _prompt_content in tqdm(user_prompt_contents):
            yield self.run_single(
                user_prompt_content=_prompt_content, **inference_kwargs
            )
