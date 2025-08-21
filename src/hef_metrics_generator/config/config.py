"""
Centralized configuration utilities for hef_metrics_generator.

This module provides helpers to resolve runtime configuration values
such as API keys, base URLs, and email identifiers required by external tools.
All values are loaded from environment variables (via dotenv).

Environment variables supported:
- PUBMED_EMAIL (required for PubMed tool)
- OPENROUTER_API_KEY (required for LLM access)
- OPENROUTER_BASE_URL (optional; defaults to https://openrouter.ai/api/v1)
- OPENROUTER_MODEL (optional; defaults to "openai/gpt-4o")

Raises:
    ConfigurationError: if a required configuration value is missing.
"""

import os
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when required configuration values are missing."""
    pass


load_dotenv()


def get_pubmed_email() -> str:
    """
    Return the PubMed email required by NCBI Entrez.

    Reads from environment variable `PUBMED_EMAIL`.

    Raises:
        ConfigurationError: if `PUBMED_EMAIL` is not set.

    Returns:
        str: PubMed contact email.
    """
    email = os.getenv("PUBMED_EMAIL")
    if not email:
        raise ConfigurationError(
            "PUBMED_EMAIL is required for PubMed tool. "
            "Set it via environment variable, e.g.: export PUBMED_EMAIL='your@email.com'"
        )
    return email


def get_openrouter_api_key() -> str:
    """
    Return the OpenRouter API key for LLM access.

    Reads from environment variable `OPENROUTER_API_KEY`.

    Raises:
        ConfigurationError: if `OPENROUTER_API_KEY` is not set.

    Returns:
        str: OpenRouter API key.
    """
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ConfigurationError(
            "OPENROUTER_API_KEY is required. "
            "Set it via environment variable, e.g.: export OPENROUTER_API_KEY='your_key_here'"
        )
    return key


def get_openrouter_base_url() -> str:
    """
    Return the OpenRouter API base URL for LLM access.

    Reads from environment variable `OPENROUTER_BASE_URL`.

    Defaults to:
        "https://openrouter.ai/api/v1"

    Returns:
        str: OpenRouter base URL.
    """
    url = os.getenv("OPENROUTER_BASE_URL")
    if not url:
        url = "https://openrouter.ai/api/v1"
    return url


def get_openrouter_model() -> str:
    """
    Return the OpenRouter model name for LLM access.

    Reads from environment variable `OPENROUTER_MODEL`.

    Defaults to:
        "openai/gpt-4o"

    Returns:
        str: model identifier.
    """
    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        model = "openai/gpt-4o"
    return model
