from typing import Literal, Pattern
import re

from mlflow_experiment.inference.postprocessing.basic_postprocessing import (
    BasicPostprocessing,
)
from mlflow_experiment.inference.postprocessing.binary_classification_output import (
    BinaryClassificationOutput,
)


class EndOfCotPostprocessing(BasicPostprocessing):
    def __init__(
        self,
        label_mapping: dict[str, Literal[0, 1]],
        end_of_cot_pattern="---",
        text_pruning_pattern: Pattern = None,
    ):
        super().__init__(label_mapping, text_pruning_pattern)
        self.end_of_cot_pattern = end_of_cot_pattern[::-1]

    def __call__(self, model_output: str) -> BinaryClassificationOutput:
        """
        Crop the model output from instance of the end_of_cot pattern onwards and then return
        the output of _run inherited from BasicPostprocessing.
        """
        try:
            a = model_output[::-1].index(self.end_of_cot_pattern)
            model_output = model_output[-(a + 1) :]
        except ValueError:
            return BinaryClassificationOutput(
                label=-1, is_wrong_format=True, is_hallucination=False
            )
        return self._run(model_output)
