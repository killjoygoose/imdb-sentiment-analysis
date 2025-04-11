"""
Some of the sklearn metrics can not be used without setting the averaging
method - so we have to wrap them to use them inside the classes of the
classification module.
"""

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score as _accuracy_score

from mlflow_experiment.inference.inference_output_df import InferenceOutputDataframe


def accuracy_score(y_true, y_pred: InferenceOutputDataframe):
    return _accuracy_score(y_true, y_pred.label)

def median_query_token_count(_, y_pred: InferenceOutputDataframe):
    return y_pred.input_token_count.quantile(0.5)

def median_response_token_count(_, y_pred: InferenceOutputDataframe):
    return y_pred.output_token_count.quantile(0.5)

def median_processing_time(_, y_pred: InferenceOutputDataframe):
    return y_pred.elapsed_time.quantile(0.5)


macro_avg_f1_score = lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
weighted_avg_f1_score = lambda y_true, y_pred: f1_score(
    y_true, y_pred, average="weighted"
)

macro_avg_precision_score = lambda y_true, y_pred: precision_score(
    y_true, y_pred, average="macro"
)
weighted_avg_precision_score = lambda y_true, y_pred: precision_score(
    y_true, y_pred, average="weighted"
)

macro_avg_recall_score = lambda y_true, y_pred: recall_score(
    y_true, y_pred, average="macro"
)
weighted_avg_recall_score = lambda y_true, y_pred: recall_score(
    y_true, y_pred, average="weighted"
)
