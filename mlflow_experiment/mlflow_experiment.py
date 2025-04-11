"""
Module for running MLflow experiments with inference and evaluation pipelines.
Logs prompts, parameters, predictions, and evaluation metrics to MLflow.
"""

import mlflow
from mlflow_experiment.evaluation.evaluation_pipeline import EvaluationPipeline
from mlflow_experiment.inference.inference_output import InferenceOutput
from mlflow_experiment.inference.inference_output_df import InferenceOutputDataframe
from mlflow_experiment.inference.inference_pipeline import InferencePipeline
import os


class MlflowExperiment:
    """
    Orchestrates MLflow experiments using inference and evaluation pipelines.

    Args:
        inference_pipeline (InferencePipeline): Pipeline for generating predictions.
        evaluation_pipeline (EvaluationPipeline): Pipeline for evaluating predictions.
        experiment_name (str): Name for the MLflow experiment.
    """

    def __init__(
        self,
        inference_pipeline: InferencePipeline,
        evaluation_pipeline: EvaluationPipeline,
        *,
        experiment_name: str = "imdb_classification",
    ):
        uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        self.evaluation_pipeline = evaluation_pipeline
        self.inference_pipeline = inference_pipeline
        self.experiment_name = experiment_name
        self.run_id = None

    def run(
        self,
        user_prompt_contents: list[dict],
        y_true: list[int],
        experiment_run_tags: dict[str, str] = None,
        description: str = None,
        run_name: str = "imdb_prompt_engineering",
        **inference_params,
    ) -> tuple[list[InferenceOutput], list[dict[str, float]]]:
        """
        Runs inference and evaluation, logs results to MLflow.

        Args:
            user_prompt_contents (list[dict]): List of input prompts for inference.
            y_true (list[int]): Ground-truth labels for evaluation.
            experiment_run_tags (dict[str, str]): Optional tags for the MLflow run, could provide
                important information e.g. about changes in the prompt.
            description (str): A free text description of the run.
            **inference_params: Additional parameters for inference.

        Returns:
            Tuple of predictions and evaluation metrics.
        """
        with mlflow.start_run(run_name=run_name, description=description) as run:
            mlflow.set_tags(experiment_run_tags)

            run._info._artifact_uri = "shit"+run.info.artifact_uri
            predictions = list(
                self.inference_pipeline.run(
                    user_prompt_contents=user_prompt_contents,
                )
            )

            mlflow.log_text(self.inference_pipeline.base_user_prompt, "user_prompt.txt")
            mlflow.log_text(self.inference_pipeline.system_prompt, "system_prompt.txt")
            mlflow.log_params(inference_params)

            evaluation_results = self.evaluation_pipeline.run(
                y_true=y_true,
                y_pred=InferenceOutputDataframe.from_inference_outputs(predictions),
            )

            mlflow.log_metrics(evaluation_results)

        return predictions, evaluation_results
