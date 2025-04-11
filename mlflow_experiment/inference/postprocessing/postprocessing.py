from abc import ABC, abstractmethod
from typing import Literal

from mlflow_experiment.inference.postprocessing.binary_classification_output import (
    BinaryClassificationOutput,
)


class Postprocessing(ABC):
    """
    Base class for all postprocessing callables.

    Args:
        label_mapping (dict[str, int]): The mapping
             from output to 0, 1
    """

    def __init__(self, label_mapping: dict[str, Literal[0, 1]]):
        self.label_mapping = label_mapping

    @abstractmethod
    def __call__(self, model_output: str) -> BinaryClassificationOutput:
        """
        Parse the model output and return a BinaryClassificationOutput object.
        """
