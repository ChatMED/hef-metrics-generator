"""
OpenAlex Tool for hef_metrics_generator.

LangChain-compatible tool for querying the OpenAlex API.
Returns structured title/URL pairs. Includes retry/backoff and standardized logging.

- Environment variable: OPENALEX_EMAIL (optional)
"""

import os
import json
import logging
import urllib.parse
from langchain.tools import tool

from hef_metrics_generator.logs.tool_query_logger import query_logger
from hef_metrics_generator.utils.http_retry import retry_request

logger = logging.getLogger("openalex_tool")


def _oa_fetch(url: str, headers: dict):
    """
    Perform an OpenAlex HTTP request and return parsed JSON dict or None on failure.
    """
    raw = retry_request(url, headers=headers)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        logger.error(f"[OpenAlex] JSON parse error: {e}")
        return None


@tool
def openalex_tool(query: str) -> list:
    """
    Search OpenAlex and return a list of works as {'title','url'} pairs.

    Args:
        query (str): Free-text search query.

    Returns:
        list[dict]: Each item has:
            - "title": Work title (str)
            - "url": OpenAlex or DOI URL (str)
    """
    query_logger.log("OpenAlex", query)

    headers = {"User-Agent": "hef_metrics_generator/0.1.0 (+https://chatmed-project.eu/)"}
    base = "https://api.openalex.org/works"

    mailto = os.getenv("OPENALEX_EMAIL")
    common = {"per-page": 30}
    if mailto:
        common["mailto"] = mailto

    params1 = dict(common)
    params1["search"] = query
    url1 = f"{base}?{urllib.parse.urlencode(params1)}"
    data = _oa_fetch(url1, headers)
    items = (data or {}).get("results", []) if data else []

    if not items:
        params2 = dict(common)
        params2["filter"] = f"title.search:{query}"
        url2 = f"{base}?{urllib.parse.urlencode(params2)}"
        data = _oa_fetch(url2, headers)
        items = (data or {}).get("results", []) if data else []

    results = []
    for it in items or []:
        title = (it.get("title") or "").strip()
        page_url = it.get("doi") or it.get("id") or ""
        if title and page_url:
            results.append({"title": title, "url": page_url})

    logger.info(f"[OpenAlex] Found {len(results)} results.")
    return results
