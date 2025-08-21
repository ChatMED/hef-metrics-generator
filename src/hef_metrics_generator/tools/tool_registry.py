"""
Tool Registry for hef_metrics_generator.

This module centralizes the registration of all LangChain-compatible tools.
It wraps each raw tool function in a `Tool` with the correct name and description,
so they can be imported and used directly by agents.
"""

from langchain_core.tools import Tool

from hef_metrics_generator.tools.arxiv_tool import arxiv_tool
from hef_metrics_generator.tools.pubmed_tool import pubmed_tool
from hef_metrics_generator.tools.ddg_tool import ddg_tool
from hef_metrics_generator.tools.semantic_scholar_tool import semantic_scholar_tool
from hef_metrics_generator.tools.openalex_tool import openalex_tool

ALL_TOOLS = [
    Tool.from_function(
        arxiv_tool,
        name="arxiv_search",
        description="Searches academic papers on arXiv for AI, ML, and scientific topics."
    ),
    Tool.from_function(
        pubmed_tool,
        name="pubmed_search",
        description="Searches biomedical and clinical literature from PubMed."
    ),
    Tool.from_function(
        ddg_tool,
        name="web_search",
        description="Performs a general-purpose web search using DuckDuckGo."
    ),
    Tool.from_function(
        semantic_scholar_tool,
        name="semantic_scholar_search",
        description="Searches research papers from Semantic Scholar."
    ),
    Tool.from_function(
        openalex_tool,
        name="openalex_search",
        description="Searches the OpenAlex database for research works, returning titles and URLs."
    )
]
