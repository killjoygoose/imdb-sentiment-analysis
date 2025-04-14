"""
Some of the sklearn metrics can not be used without setting the averaging
method - so we have to wrap them to use them inside the classes of the
classification module.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score as _accuracy_score,
)

from mlflow_experiment.inference.inference_output_df import InferenceOutputDataframe


"""
Classification metrics
"""


def accuracy_score(y_true: pd.Series, y_pred: InferenceOutputDataframe):
    return _accuracy_score(y_true, y_pred.label)


def false_positive_rate(y_true: pd.Series, y_pred: InferenceOutputDataframe):
    # positive is 0! (hallucinations not included!)
    return len(y_pred[(y_pred.label == 0) & (y_true == 1)]) / (y_true == 1).sum()


def precision(y_true: pd.Series, y_pred: InferenceOutputDataframe):
    # positive is 0! (hallucinations not included!)
    return len(y_pred[(y_pred.label == 0) & (y_true == 0)]) / (y_pred.label == 0).sum()


def recall(y_true: pd.Series, y_pred: InferenceOutputDataframe):
    # positive is 0! (hallucinations not included!)
    return len(y_pred[(y_pred.label == 0) & (y_true == 0)]) / (y_true == 0).sum()


def false_negative_rate(y_true: pd.Series, y_pred: InferenceOutputDataframe):
    return 1 - recall(y_true, y_pred)


"""
Query efficiency metrics
"""


def median_query_token_count(_, y_pred: InferenceOutputDataframe):
    return y_pred.input_token_count.quantile(0.5)


def median_response_token_count(_, y_pred: InferenceOutputDataframe):
    return y_pred.output_token_count.quantile(0.5)


"""
Runtime efficiency metrics
"""


def median_processing_time(_, y_pred: InferenceOutputDataframe):
    return y_pred.elapsed_time.quantile(0.5)


def median_tokens_per_second(_, y_pred: InferenceOutputDataframe):
    return (
        (y_pred.output_token_count + y_pred.output_token_count)
        / (y_pred.elapsed_time + 1e-9)
    ).quantile(0.5)


"""
Output text quality metrics
"""


def hallucination_rate(_, y_pred: InferenceOutputDataframe):
    return y_pred.is_hallucination.sum() / len(y_pred)


def bad_output_format_rate(_, y_pred: InferenceOutputDataframe):
    return y_pred.is_wrong_format.sum() / len(y_pred)
