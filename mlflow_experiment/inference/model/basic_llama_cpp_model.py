from typing import Any

from mlflow_experiment.inference.model.llama_cpp_model import LlamaCppModel
from mlflow_experiment.inference.model.llama_cpp_model_output import LlamaCppModelOutput


class BasicLlamaCppModel(LlamaCppModel):
    """
    A basic wrapper around Llama that builds the user prompt.
    """

    def run(
        self,
        system_prompt: str,
        user_prompt_content: dict[str, Any],
        **inference_kwargs
    ) -> LlamaCppModelOutput:
        user_prompt = self.user_prompt_builder.build_user_prompt(user_prompt_content)

        output = self.model.create_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **inference_kwargs
        )

        return LlamaCppModelOutput(
            content=output["choices"][0]["message"]["content"],
            prompt_tokens=output["usage"]["prompt_tokens"],
            completion_tokens=output["usage"]["completion_tokens"],
        )
