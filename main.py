"""
This is a small executable whose aim is to let you run inference / evaluation on
a jsonl file in the same format as the test data. It is not a ubiquitous tool
and is not designed for commercial use. Prompt is stored inside the prompt dir
so that one won't have to start mlflow just to test this script.
"""

import argparse
import json
import sys
from pathlib import Path
from llama_cpp import Llama
import pandas as pd
from eda_utils.helpers import get_gpu_memory
from mlflow_experiment import EvaluationPipeline, InferencePipeline
from mlflow_experiment.evaluation.metrics import (
    accuracy_score,
    precision,
    recall,
    false_negative_rate,
    false_positive_rate,
    median_response_token_count,
    median_query_token_count,
    median_processing_time,
    median_tokens_per_second,
    hallucination_rate,
    bad_output_format_rate,
)
from mlflow_experiment.inference.inference_output_df import InferenceOutputDataframe
from mlflow_experiment.inference.postprocessing.basic_postprocessing import (
    BasicPostprocessing,
)
from mlflow_experiment.inference.prompt_building.basic_user_prompt_builder import (
    BasicUserPromptBuilder,
)
from mlflow_experiment.inference.model.basic_llama_cpp_model import BasicLlamaCppModel
import logging

_LOGGER = logging.getLogger("my_logger")
_LOGGER.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
_LOGGER.addHandler(handler)

if __name__ == "__main__":
    PARAMETERS_SIZE = 1.31e9
    N_LAYERS = 28  # see https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
    N_BYTES_PER_LAYER = int(PARAMETERS_SIZE / 28) * (5.0 / 8)  # bytes per layer

    PROJECT_FOLDER = Path(__file__).parent
    MODEL_NAME = "Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"
    MODEL_PATH = (PROJECT_FOLDER / "models") / MODEL_NAME

    parser = argparse.ArgumentParser(description="Read input and output file paths.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input file. "
        "Must be in the jsonl format and contain a review column. "
        "If run_evaluation is set, then a 'label' column must also "
        "be available in the file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Name of the output file to create. "
        "Ideally the extension should be .jsonl, as we are "
        "saving the results in jsonl. If run_evaluation is also "
        "created an additional ",
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Whether to run evaluation or only inference.",
        default=False,
    )

    parser.add_argument(
        "--run_on_cpu",
        action="store_true",
        help="Whether to run the model on cpu, if there is a gpu detected. "
        "Defaults to false.",
        default=False,
    )

    args = parser.parse_args()
    output_file = Path(args.output_file)
    gpu_memory_stats = get_gpu_memory()

    if args.run_on_cpu or gpu_memory_stats.empty:
        _LOGGER.info(f"RUNNING ON CPU!")

        model = Llama(
            str(MODEL_PATH), verbose=False, n_ctx=2000, n_gpu_layers=0, n_batch=2000
        )
    else:
        largest_free_memory_location = gpu_memory_stats["free_MB"].argmax()
        free_gpu_memory_size = gpu_memory_stats["free_MB"][
            largest_free_memory_location
        ] * (
            10**6
        )  # convert to bytes
        n_layers = int((free_gpu_memory_size / N_BYTES_PER_LAYER) * 0.65)

        _LOGGER.info(f"RUNNING ON GPU WITH NUMBER OF LAYERS = {n_layers}")
        model = Llama(
            str(MODEL_PATH),
            verbose=False,
            n_ctx=2000,
            n_gpu_layers=n_layers,
            n_batch=2000,
        )

    with open((PROJECT_FOLDER / "prompt") / "user_prompt.txt", "r") as f:
        prompt = f.read()

    try:
        data = pd.read_json(args.input_file, lines=True).iloc[:10]
    except ValueError as e:
        raise ValueError("The input file must point to a valid jsonl file!") from e

    if "review" not in data.columns:
        raise ValueError("The data must contain a 'review' column!")

    if (
        args.run_evaluation
        and "label" not in data.columns
        and data["label"].unique().astype(int).tolist() != [0, 1]
    ):
        raise ValueError(
            "The data must contain a 'label' column, containing binary numeric labels 0 and 1!"
        )

    prompt_builder = BasicUserPromptBuilder(prompt)
    llama_model = BasicLlamaCppModel(
        model, user_prompt_builder=prompt_builder, model_name=MODEL_NAME
    )

    postprocessing_fn = BasicPostprocessing({"negative": 1, "positive": 0})

    inference_pipeline = InferencePipeline(
        llama_model,
        system_prompt="You are a sentiment analysis system. Your goal is to categorize movie reviews into positive or negative. Follow the instructions given precisely!",
        postprocessing_fn=postprocessing_fn,
    )

    predictions = InferenceOutputDataframe.from_inference_outputs(
        inference_pipeline.run(
            user_prompt_contents=data[["review"]].to_dict(orient="records"),
        )
    )

    if args.run_evaluation:
        evaluation_pipeline = EvaluationPipeline(
            accuracy_score,
            precision,
            recall,
            false_negative_rate,
            false_positive_rate,
            median_response_token_count,
            median_query_token_count,
            median_processing_time,
            median_tokens_per_second,
            hallucination_rate,
            bad_output_format_rate,
        )
        evaluation_results = evaluation_pipeline.run(
            y_true=data.label,
            y_pred=predictions,
        )
        eval_output_file = output_file.parent / (output_file.stem + "_evaluations.json")
        _LOGGER.info("Saving model outputs to: {}".format(output_file))
        with open(eval_output_file, "w") as f:
            json.dump(evaluation_results, f)

    _LOGGER.info("Saving model outputs to: {}".format(output_file))
    predictions.to_json(output_file, lines=True, orient="records")
