"""
Output metric schema for hef_metrics_generator.
Defines Source and Metric with strict validation.
"""

from __future__ import annotations
from typing import List, Literal, Tuple
from pydantic import BaseModel, HttpUrl, field_validator, model_validator, ConfigDict
from urllib.parse import urlparse
import re

from hef_metrics_generator.utils.constants import MAX_METRIC_NAME_LEN, TRUSTED_DOMAINS

_ALPHA_WORDS_RE = re.compile(r"^[A-Za-z ]+$")
_METRIC_NAME_RE = re.compile(r"^[A-Za-z0-9 \-\(\)_/]+$")


def validate_alpha_words(value: str) -> str:
    """
    Accept only letters and spaces (no digits, punctuation, or symbols).
    Used for task labels (domain/field/type).
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("must be a non-empty string")
    if len(v) < 2 or len(v) > 100:
        raise ValueError("must be between 2 and 100 characters")
    if not _ALPHA_WORDS_RE.fullmatch(v):
        raise ValueError("must contain only letters and spaces (no digits or punctuation)")
    return v


def validate_metric_name(value: str) -> str:
    """
    Allow realistic metric names (e.g., F1, BLEU-4, Knowledge Consistency (KC)).
    Letters/digits/spaces plus: - _ / ( )
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("metric name must be a non-empty string")
    if len(v) > MAX_METRIC_NAME_LEN:
        raise ValueError(f"metric name is too long (>{MAX_METRIC_NAME_LEN} chars)")
    if not _METRIC_NAME_RE.fullmatch(v):
        raise ValueError("metric name contains invalid characters")
    return v


def validate_general_text(value: str, max_len: int = 500) -> str:
    """
    General text sanity:
    - non-empty after strip
    - contains at least one letter
    - not absurdly long (default 500 chars)
    Allows punctuation/digits (needed for paper titles, descriptions).
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("must be non-empty")
    if len(v) > max_len:
        raise ValueError(f"is unreasonably long (>{max_len} chars)")
    if not re.search(r"[A-Za-z]", v):
        raise ValueError("must contain at least one letter")
    return v


class Source(BaseModel):
    title: str
    url: HttpUrl

    @field_validator("title")
    @classmethod
    def _validate_title(cls, v: str) -> str:
        return validate_general_text(v, max_len=300)


class Metric(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    metric: str
    min: Literal[0, 1]
    max: Literal[1, 5]
    description: str
    relevance: str
    sources: List[Source]
    search_queries: List[str]

    @field_validator("metric")
    @classmethod
    def _validate_metric_name(cls, v: str) -> str:
        return validate_metric_name(v)

    @field_validator("description", "relevance")
    @classmethod
    def _validate_text_fields(cls, v: str) -> str:
        return validate_general_text(v, max_len=500)

    @field_validator("search_queries")
    @classmethod
    def _validate_queries(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("search_queries must not be empty")
        cleaned = []
        for q in v:
            q = (q or "").strip()
            if q:
                cleaned.append(q)
        if not cleaned:
            raise ValueError("search_queries must not be empty")
        return cleaned

    @field_validator("sources")
    @classmethod
    def _filter_sources(cls, v: List[Source]) -> List[Source]:
        """
        Keep only sources from trusted domains.
        Drop untrusted ones silently.
        """
        trusted = []
        for s in v:
            host = urlparse(str(s.url)).netloc
            if any(domain in host for domain in TRUSTED_DOMAINS):
                trusted.append(s)
        if not trusted:
            raise ValueError("no valid sources left after filtering untrusted domains")
        return trusted

    @model_validator(mode="after")
    def _validate_min_max_combo(self) -> "Metric":
        combo: Tuple[int, int] = (self.min, self.max)
        if combo not in {(1, 5), (0, 1)}:
            raise ValueError("min/max must be either (1,5) or (0,1)")
        return self
