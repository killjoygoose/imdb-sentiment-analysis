from multiprocessing import Pool, cpu_count
import spacy
from nltk.stem import SnowballStemmer
from itertools import chain
import logging

from eda_utils.helpers import split_in_batches

_LOGGER = logging.getLogger(__name__)
try:
    _NLP = spacy.load("en_core_web_lg", disable=["ner", "parser", "lemmatizer"])
except OSError:
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    _NLP = spacy.load("en_core_web_lg")

_STEMMER = SnowballStemmer("english")


class PreprocessingPipelineEDA:

    def __init__(self, allowed_pos_tags: set[str]):
        self.allowed_pos_tags = allowed_pos_tags
        self.n_cpus = cpu_count()

    def filter_by_pos_tag(self, text: str) -> list[str]:
        """
        Tokenize a string, filtering the tokens that are not
        within the allowed POS tags.
        """
        doc = _NLP(text)
        tokens = [
            token.text.lower()
            for token in doc
            if token.pos_ in self.allowed_pos_tags and token.is_alpha
        ]
        return tokens

    def _stem_tokens(self, tokens: list[str]):
        """
        Stem each of the elements of a list of strings.
        """
        return [_STEMMER.stem(t) for t in tokens]

    def _whitespace_concat(self, tokens: list[str]):
        return " ".join(tokens)

    def process_single(self, text: str) -> str:
        """
        Apply the full pipeline on a single string.
        """
        return self._whitespace_concat(self._stem_tokens(self.filter_by_pos_tag(text)))

    def process_bulk(self, corpus: list[str]) -> list[str]:
        output = []
        for text in corpus:
            output.append(self.process_single(text))
        return output

    def multiprocessed_processing(self, corpus: list[str], n_workers=-1) -> list[str]:
        """
        Split the corpus in n_workers chunks sizes and process them in parallel.
        """
        n_workers = n_workers if n_workers >= 1 else self.n_cpus

        if len(corpus) <= n_workers:
            _LOGGER.info(
                "Size of the corpus is less than the number of workers. "
                "Processing on single core."
            )
            return self.process_bulk(corpus)

        batch_size = max(len(corpus) // n_workers, 1)
        batches = split_in_batches(corpus, batch_size=batch_size)
        _LOGGER.info(f"Created {len(batches)} of sizes {[len(i) for i in batches]}")
        with Pool(processes=n_workers) as pool:
            processed = pool.map(self.process_bulk, batches)

        return list(chain(*processed))
