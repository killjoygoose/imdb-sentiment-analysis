from llama_cpp import Llama
import pandas as pd
from langdetect import detect
import dotenv
dotenv.load_dotenv(".env", override=True)

MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf"
model = Llama(MODEL_PATH, verbose=False, n_ctx=32768)

# a = model.create_chat_completion(messages=[
#     {"role": "system",
#      "content": "You are a sentiment analysis sytem. Your task is to say whether the user review is positive or negative!"},
#     {"role": "user", "content": """You are given the following review:
#
# I have seen this film only once, on TV, and it has not been repeated. This is strange when you consider the rubbish that is repeated over and over again. Usually horror movies for me are a source of amusement, but this one really scared me.<br /><br />DO NOT READ THE NEXT BIT IF YOU HAVE'NT SEEN THE FILM YET<br /><br />The scariest bit is when the townsfolk pursue the preacher to where his wife lies almost dead (they'd been poisoning her). He asks who the hell are you people anyway. One by one they give their true identities. The girl who was pretending to be deaf in order to corrupt and seduce him says 'I am Lilith, the witch who loved Adam before Eve'.
#
#      --------
#      Your task: find the places where the user gives an evaluation to the movie. Possible, but not exclusive, topics for evaluation can be actors' performance level, overall storry, etc.
#
#      ----
#      Output format: IN A SINGLE WORD: Positive or Negative : describe the overall sentiment!
#      """}
# ], logprobs=True)

from mlflow_experiment import MlflowExperiment, EvaluationPipeline, InferencePipeline
from mlflow_experiment.evaluation.metrics import accuracy_score, median_response_token_count, median_query_token_count, median_processing_time
from mlflow_experiment.inference.prompt_building.basic_user_prompt_builder import (
    BasicUserPromptBuilder,
)
from mlflow_experiment.inference.postprocessing.basic_postprocessing import (
    BasicPostprocessing,
)

ev_pipeline = EvaluationPipeline({"accuracy": accuracy_score,
                                  "median_response_token_count": median_response_token_count,
                                  "median_query_token_count": median_query_token_count,
                                  "median_processing_time": median_processing_time
                                  })
prompt_builder = BasicUserPromptBuilder(
    "Output randomly either 1 or 0. Nothing more - just one of these numbers! Your output must be either '1' or '0'!"
)
postprocessing_fn = BasicPostprocessing({"1": 1, "0": 0})
inf_pipeline = InferencePipeline(
    model,
    system_prompt="You are a random binary number generator!!! Just output 1 or 0. Make sure you randomly chose and don't always output just one of them!",
    user_prompt_builder=prompt_builder,
    postprocessing_fn=postprocessing_fn,
)

exp = MlflowExperiment(
    inference_pipeline=inf_pipeline,
    evaluation_pipeline=ev_pipeline,
    experiment_name="hj",
)

exp.run(
    [{}, {}, {}, {}, {}],
    y_true=[0, 0, 1, 1, 0],
    experiment_run_tags={"justification": "just a test"},
)

model._sampler.close()
model.close()