"""
Semantic Scholar Tool for hef_metrics_generator.

LangChain-compatible tool for querying Semantic Scholar.
Returns structured title/URL pairs. Includes retry/backoff and standardized logging.

- Environment variable: SEMANTIC_SCHOLAR_API_KEY (optional)
"""

import os
import json
import logging
import urllib.parse
from langchain.tools import tool

from hef_metrics_generator.logs.tool_query_logger import query_logger
from hef_metrics_generator.utils.http_retry import retry_request

logger = logging.getLogger("semantic_scholar_tool")


@tool
def semantic_scholar_tool(query: str) -> list:
    """
    Search Semantic Scholar and return a list of papers as {'title','url'} pairs.

    Args:
        query (str): Free-text search query.

    Returns:
        list[dict]: Each item has:
            - "title": Paper title (str)
            - "url": Semantic Scholar page URL (str)
    """
    query_logger.log("SemanticScholar", query)

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 30,
        "fields": "title,url,paperId,year,openAccessPdf",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    headers = {
        "User-Agent": "hef_metrics_generator/0.1.0 (+https://chatmed-project.eu/)"
    }
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    raw = retry_request(url, headers=headers, retries=2, backoff=2.0)
    if not raw:
        logger.error("[SemanticScholar] Error: no response after retries.")
        return []

    try:
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"[SemanticScholar] JSON parse error: {e}")
        return []

    items = data.get("data", []) or []
    results = []
    for it in items:
        title = (it.get("title") or "").strip()
        page_url = it.get("url")
        if not page_url:
            paper_id = it.get("paperId")
            if paper_id:
                page_url = f"https://www.semanticscholar.org/paper/{paper_id}"
            else:
                page_url = ""

        if title and page_url:
            results.append({"title": title, "url": page_url})

    logger.info(f"[SemanticScholar] Found {len(results)} results.")
    return results
