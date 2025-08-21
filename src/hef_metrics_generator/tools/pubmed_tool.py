"""
PubMed Tool for hef_metrics_generator.

LangChain-compatible tool for querying PubMed (via Biopython Entrez).
Returns structured title/URL pairs. Includes retry/backoff and standardized logging.

- Environment variable: PUBMED_EMAIL (required)
"""

import logging
import time
from langchain.tools import tool
from Bio import Entrez, Medline

from hef_metrics_generator.logs.tool_query_logger import query_logger
from hef_metrics_generator.config.config import get_pubmed_email, ConfigurationError

logger = logging.getLogger("pubmed_tool")


def _entrez_with_retry(func, retries: int = 2, backoff: float = 2.0, **kwargs):
    """
    Wrap Entrez functions with retry on transient errors.
    """
    for attempt in range(retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt < retries:
                sleep_s = backoff * (2 ** attempt)
                logger.warning(f"[PubMed] Error: {e}; retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            logger.error(f"[PubMed] Error: {e}")
            return None
    return None


@tool
def pubmed_tool(query: str) -> list:
    """
    Search PubMed for a given query and return a list of results.

    Args:
        query (str): The search term to query PubMed.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - "title": The article title (str)
            - "url": Link to the PubMed article (str)
    """
    query_logger.log("PubMed", query)
    try:
        Entrez.email = get_pubmed_email()

        handle = _entrez_with_retry(Entrez.esearch, db="pubmed", term=query, retmax=30)
        if not handle:
            return []
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        if not ids:
            return []

        handle = _entrez_with_retry(
            Entrez.efetch,
            db="pubmed",
            id=",".join(ids),
            rettype="medline",
            retmode="text"
        )
        if not handle:
            return []

        records = Medline.parse(handle)
        entries = []
        for rec in records:
            pmid = rec.get("PMID", "")
            title = rec.get("TI", "No title available")
            entries.append({
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        logger.info(f"[PubMed] Found {len(entries)} results.")
        return entries

    except ConfigurationError as e:
        logger.error(f"[PubMed] Config error: {e}")
        raise
    except Exception as e:
        logger.error(f"[PubMed] Error: {e}")
        return []
