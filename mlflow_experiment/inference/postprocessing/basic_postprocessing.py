from typing import Literal, Pattern
import re
from mlflow_experiment.inference.postprocessing.binary_classification_output import (
    BinaryClassificationOutput,
)
from mlflow_experiment.inference.postprocessing.postprocessing import Postprocessing


class BasicPostprocessing(Postprocessing):
    def __init__(
        self,
        label_mapping: dict[str, Literal[0, 1]],
        text_pruning_pattern: Pattern = None,
    ):
        super().__init__(label_mapping)
        self.text_pruning_pattern = text_pruning_pattern or re.compile(r"[^a-zA-Z0-9]")

    def _run(self, model_output: str) -> BinaryClassificationOutput:
        model_output = self.text_pruning_pattern.sub(" ", model_output).strip()
        if len(model_output.split()) > 1:
            return BinaryClassificationOutput(
                label=-1, is_wrong_format=True, is_hallucination=False
            )
        if model_output not in self.label_mapping:
            return BinaryClassificationOutput(
                label=-1, is_wrong_format=False, is_hallucination=True
            )
        return BinaryClassificationOutput(
            label=self.label_mapping[model_output],
            is_wrong_format=False,
            is_hallucination=False,
        )

    def __call__(self, model_output: str) -> BinaryClassificationOutput:
        return self._run(model_output)
