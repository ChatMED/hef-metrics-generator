"""
Tool output validation for hef_metrics_generator.
Uses LLM assistance to check whether sources are relevant.
"""

import logging
from typing import Optional
from langchain_core.language_models import BaseLanguageModel

from hef_metrics_generator.schemas.metric import Source

logger = logging.getLogger("tool_validator")


def validate_source_with_llm(source: Source, llm: BaseLanguageModel, task_domain: str) -> Optional[Source]:
    """
    Validate a single Source object using the LLM.

    Args:
        source (Source): The source to validate.
        llm (BaseLanguageModel): LLM instance used for relevance checking.
        task_domain (str): The domain context (e.g., "healthcare").

    Returns:
        Optional[Source]: The validated source if relevant, else None.
    """
    prompt = f"""
    You are filtering research papers to decide whether they are relevant
    for supporting human evaluation metrics of large language models (LLMs)
    in the domain: {task_domain}.

    Here is a candidate source:
    Title: {source.title}
    URL: {source.url}

    Does this source discuss the evaluation of LLMs or AI-generated content
    in {task_domain} settings?

    Answer only with "yes" or "no".
    """

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip().lower()

        if answer.startswith("yes"):
            logger.info(f"[Validator] ACCEPTED source: {source.title} ({source.url})")
            return source
        else:
            logger.warning(f"[Validator] DROPPED source: {source.title} ({source.url})")
            return None

    except Exception as e:
        logger.error(f"[Validator] LLM validation error for source '{source.title}': {e}")
        return None
