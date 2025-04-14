from time import time
from typing import Callable, Any
import logging

import pandas as pd

from mlflow_experiment.inference.inference_output_df import InferenceOutputDataframe

_LOGGER = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(
        self,
        *evaluation_functions: Callable[[pd.Series, InferenceOutputDataframe], float],
    ):
        self.evaluation_functions = {i.__name__: i for i in evaluation_functions}

    def run(
        self,
        y_true: list[Any],
        y_pred: InferenceOutputDataframe,
    ) -> dict[str, float]:
        results = {}

        for eval_func_name, eval_func in self.evaluation_functions.items():
            _start = time()

            _LOGGER.debug("Computing %s.", eval_func_name)
            results[eval_func_name] = eval_func(pd.Series(y_true), y_pred)

            _LOGGER.debug(
                "Successfully computed %s: %.3f for %.3fs. Logging to Mlflow...",
                eval_func_name,
                results[eval_func_name],
                (time() - _start),
            )

        return results
