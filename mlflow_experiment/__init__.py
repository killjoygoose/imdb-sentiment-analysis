"""
Package containing abstractions that facilitate prompt-engineering
projects by providing abstractions for automated inference and logging of results
"""

from mlflow_experiment.mlflow_experiment import MlflowExperiment
from mlflow_experiment.inference.inference_pipeline import InferencePipeline
from mlflow_experiment.evaluation.evaluation_pipeline import EvaluationPipeline
