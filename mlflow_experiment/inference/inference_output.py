"""
Defines the `InferenceOutput` dataclass for storing results of a single inference.
"""

from typing import Literal
from dataclasses import dataclass, asdict


@dataclass
class InferenceOutput:
    output_string: str
    label: Literal[0, 1, -1]  # -1 for fail
    is_hallucination: bool
    is_wrong_format: bool  # when output could not be parsed
    input_token_count: int
    output_token_count: int
    elapsed_time: float

    def asdict(self):
        return asdict(self)
