"""
Project-wide constants for hef_metrics_generator.
Keep schema defaults and shared limits here to avoid drift.
"""

NUM_METRICS_DEFAULT: int = 10
MIN_SOURCES_DEFAULT: int = 3

MAX_METRIC_NAME_LEN: int = 100

TRUSTED_DOMAINS = (
    "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org",
    "semanticscholar.org",
    "springer.com",
    "sciencedirect.com",
    "wiley.com",
    "jamanetwork.com",
    "nejm.org",
    "thelancet.com",
    "bmcmededuc.biomedcentral.com",
    "mednexus.org",
    "doi.org",
)