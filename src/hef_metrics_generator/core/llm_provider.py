"""
LLM provider for hef_metrics_generator.

This module exposes a single factory: `get_llm()`, which returns a LangChain
`ChatOpenAI` client configured to talk to OpenRouter (default) or any
OpenAI-compatible endpoint.

Configuration is centralized in hef_metrics_generator.config.config.

Optional overrides can be passed as function args:
    - temperature (float): sampling temperature. Defaults to 0.5.
    - max_tokens (int): maximum tokens to generate. Defaults to 10000.

Raises:
    ConfigurationError: if API key or base URL cannot be resolved.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from hef_metrics_generator.config.config import (
    get_openrouter_api_key,
    get_openrouter_base_url,
    get_openrouter_model,
)

load_dotenv()


def _get_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable among `names`, or default."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


def get_llm(
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Construct and return a LangChain `ChatOpenAI` client configured via
    hef_metrics_generator.config.config.

    Args:
        temperature (float, optional): sampling temperature. Defaults to 0.5.
        max_tokens (int, optional): maximum tokens for the response. Defaults to 10000.

    Returns:
        ChatOpenAI: a LangChain-compatible chat LLM client.

    Raises:
        ConfigurationError: if required environment variables are missing.
    """
    api_key = get_openrouter_api_key()
    base_url = get_openrouter_base_url()
    model_name = get_openrouter_model()

    temperature = 0.5 if temperature is None else temperature
    max_tokens = 3000 if max_tokens is None else max_tokens

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
