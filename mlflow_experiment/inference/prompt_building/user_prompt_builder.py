"""
Base class
"""

import re
from abc import ABC, abstractmethod
from typing import Any
import logging

_LOGGER = logging.getLogger(__name__)


class UserPromptBuilder(ABC):
    """
    Base class for all user prompt builders, whose responsibility is
    to insert the specific user prompt content into the base user prompt.
    There might be different approaches to inserting the user prompt content
    - e.g. for RAG a query to a database will be required and the output of the
    RAG will be part of the final user prompt.
    """

    __regex = re.compile(
        r"(?<=[^{]\{)(\w+)(?=}[^}])|(?<=\{)(\w+)(?=}[^}])|(?<=[^{]\{)(\w+)(?=})"
    )

    def __init__(self, base_user_prompt: str):
        self.base_user_prompt = base_user_prompt

    @abstractmethod
    def _build_user_prompt(self, user_prompt_content: dict[str, Any] = None) -> str:
        """
        Function to build the user prompt given an optional user prompt content
        possibly using the base_user_prompt.
        """

    def build_user_prompt(self, user_prompt_content: dict[str, Any] = None) -> str:
        """
        Validate the prompt content and build the final user prompt.

        Args:
            user_prompt_content: (dict[str, Any]) The contents (format fields) which to fill the
                user prompt with.
        """
        self._validate_prompt_content(user_prompt_content)
        return self._build_user_prompt(user_prompt_content)

    def _validate_prompt_content(
        self,
        user_prompt_content: dict,
    ):
        """
        Check if there is any discrepancy between the format fields of the provided user_prompt and
        the keys of the user_prompt_content.

        Args:
            user_prompt_content (dict): A dict holding the datapoint-specific information
                to be inserted into the user_prompt.


        Raises:
            KeyError: if raise_if_inconsistent is set, then KeyError is raised if
                the prompt does not contain the same keys as the content.
                Else just a warning is raised.
        """
        user_prompt_keys = set(self.__regex.findall(self.base_user_prompt))
        content_keys = set(user_prompt_content.keys())
        missing_in_prompt = content_keys.difference(user_prompt_keys)
        missing_in_content = user_prompt_keys.difference(content_keys)
        error_message = (
            ""
            + (
                f"user_prompt_content contains the following format fields that "
                f"are not contained in the user_prompt: {missing_in_prompt} "
            )
            * (len(missing_in_prompt) > 0)
            + (
                f"user_prompt contains the following format fields that are not "
                f"contained in the user_prompt_content: {missing_in_content} "
            )
            * (len(missing_in_content) > 0)
        )
        if error_message:
            raise KeyError(error_message)
