from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class BinaryClassificationOutput:
    label: Literal[0, 1, -1]  # -1 for fail
    is_hallucination: bool
    is_wrong_format: bool  # when output could not be parsed
