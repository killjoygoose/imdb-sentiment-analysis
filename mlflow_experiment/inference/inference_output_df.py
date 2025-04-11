from typing import Self
import pandas as pd

from mlflow_experiment.inference.inference_output import InferenceOutput


class InferenceOutputDataframe:
    """
    A wrapper around a dataframe object which guarantees that
    the object has particular columns.
    """

    def __init__(self, data: pd.DataFrame):
        self.validate_data(data)
        self.data = data

    def validate_data(self, data: pd.DataFrame):
        required_properties = [
            name
            for name, object in self.__class__.__dict__.items()
            if isinstance(object, property)
        ]
        if diff := set(required_properties).difference(data.columns):
            raise ValueError(
                "data is not a valid input as the following columns are missing,"
                f"but are required: {diff}"
            )

    @property
    def output_string(self) -> pd.Series:
        return self.data.output_string

    @property
    def label(self) -> pd.Series:
        return self.data.label

    @property
    def is_hallucination(self) -> pd.Series:
        return self.data.is_hallucination

    @property
    def is_wrong_format(self) -> pd.Series:
        return self.data.is_wrong_format

    @property
    def input_token_count(self) -> pd.Series:
        return self.data.input_token_count

    @property
    def output_token_count(self) -> pd.Series:
        return self.data.output_token_count

    @property
    def elapsed_time(self) -> pd.Series:
        return self.data.elapsed_time

    def __getattr__(self, item):
        return getattr(self.data, item)

    @classmethod
    def from_inference_outputs(cls, inference_outputs: list[InferenceOutput]) -> Self:
        return pd.DataFrame([i.asdict() for i in inference_outputs])
