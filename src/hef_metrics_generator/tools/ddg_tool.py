"""
DuckDuckGo Tool for hef_metrics_generator.

LangChain-compatible tool for querying DuckDuckGo.
Returns structured title/URL pairs. Includes retry/backoff and standardized logging.
"""

import logging
import time
from langchain.tools import tool
from duckduckgo_search import DDGS

from hef_metrics_generator.logs.tool_query_logger import query_logger

logger = logging.getLogger("ddg_tool")


def _ddg_with_retry(query: str, retries: int = 2, backoff: float = 2.0):
    """
    Run DuckDuckGo text search with retry + exponential backoff.

    Returns raw DDG results list (or []).
    """
    for attempt in range(retries + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    region="wt-wt",
                    safesearch="off",
                    max_results=100
                )
                return results or []
        except Exception as e:
            if attempt < retries:
                sleep_s = backoff * (2 ** attempt)
                logger.warning(f"[DuckDuckGo] Error: {e}; retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            logger.error(f"[DuckDuckGo] Error: {e}")
            return []
    return []


@tool
def ddg_tool(query: str) -> list:
    """
    Search DuckDuckGo for a given query and return a list of results.

    Args:
        query (str): The search term to query DuckDuckGo.

    Returns:
        list[dict]: Each dict contains:
            - "title": Result title (str)
            - "url": Link to the result (str)
    """
    query_logger.log("DuckDuckGo", query)

    raw_results = _ddg_with_retry(query)
    formatted = []
    for r in raw_results:
        formatted.append({
            "title": r.get("title", "No Title"),
            "url": r.get("href") or r.get("link") or ""
        })

    logger.info(f"[DuckDuckGo] Found {len(formatted)} results.")
    return formatted
