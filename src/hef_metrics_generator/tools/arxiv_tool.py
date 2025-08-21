"""
ArXiv Tool for hef_metrics_generator.

LangChain-compatible tool for querying the ArXiv API (Atom feed).
Returns structured title/URL pairs. Includes retry/backoff and standardized logging.
"""

import logging
import feedparser
from langchain.tools import tool

from hef_metrics_generator.logs.tool_query_logger import query_logger
from hef_metrics_generator.utils.http_retry import retry_request

logger = logging.getLogger("arxiv_tool")


@tool
def arxiv_tool(query: str) -> list:
    """
    Search ArXiv for a given query and return a list of results.

    Args:
        query (str): The search term to query ArXiv.

    Returns:
        list[dict]: Each dict contains:
            - "title": Paper title (str)
            - "url": Link to the paper on ArXiv (str)
    """
    query_logger.log("ArXiv", query)
    base_url = "http://export.arxiv.org/api/query"
    url = f"{base_url}?search_query={query.replace(' ', '+')}&start=0&max_results=30"

    try:
        raw = retry_request(url)
        if not raw:
            logger.error("[ArXiv] Error: no response after retries.")
            return []

        feed = feedparser.parse(raw)
    except Exception as e:
        logger.error(f"[ArXiv] Error parsing feed: {e}")
        return []

    results = []
    for entry in getattr(feed, "entries", []):
        results.append({
            "title": entry.title.strip().replace("\n", " "),
            "url": entry.id
        })

    logger.info(f"[ArXiv] Found {len(results)} results.")
    return results
