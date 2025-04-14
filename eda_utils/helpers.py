from math import floor
import logging
import pandas as pd
import re
import subprocess
from typing import Any

_LOGGER = logging.getLogger()


def split_in_batches(corpus: list[Any], batch_size: int) -> list[Any]:
    return [corpus[i : i + batch_size] for i in range(0, len(corpus), batch_size)]


def keep_beginning_and_ending(
    text: str, part_beginning: int | float, part_ending: int | float
) -> str:
    """
    Discard the middle - keep only n tokens from the beginning and n tokens to the end.

    Args:
        text: The text.
        part_beginning (int | float): N words, whitespace tokenized, or a
            percentage of the words in the text, to keep from the beginning.
        part_ending (int | float): N words, whitespace tokenized, or a
            percentage of the words in the text, to keep to the end.

    Returns:
        The cropped text.
    """
    tokens = text.split()
    if isinstance(part_beginning, float):
        part_beginning = floor(len(tokens) * part_beginning)

    if isinstance(part_ending, float):
        part_ending = floor(len(tokens) * part_ending)
    out = " ".join(tokens[:part_beginning] + tokens[-part_ending:])
    return out


def get_gpu_memory():
    """
    Check if cuda is available for gpu inference.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free,memory.used",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        output = result.stdout.strip().split("\n")
        gpus = []
        for line in output:
            total, free, used = map(int, re.findall(r"\d+", line))
            gpus.append(
                {
                    "total_MB": total,
                    "free_MB": free,
                    "used_MB": used,
                }
            )
        return pd.DataFrame(gpus)
    except Exception as e:
        _LOGGER.info(f"Error querying GPU memory: {e}")
        return pd.DataFrame([])
