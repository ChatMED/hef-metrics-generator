"""
Task context schema for hef_metrics_generator.
Carries task domain/field/type and generation controls (num_metrics, min_sources_per_metric).
"""

from __future__ import annotations
from pydantic import BaseModel, field_validator
import re

from hef_metrics_generator.utils.constants import NUM_METRICS_DEFAULT, MIN_SOURCES_DEFAULT

_ALPHA_WORDS_RE = re.compile(r"^[A-Za-z ]+$")


def validate_alpha_words(value: str) -> str:
    """
    Accept only letters and spaces (no digits, punctuation, or symbols).
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("must be a non-empty string")
    if len(v) < 2 or len(v) > 100:
        raise ValueError("must be between 2 and 100 characters")
    if not _ALPHA_WORDS_RE.fullmatch(v):
        raise ValueError("must contain only letters and spaces (no digits or punctuation)")
    return v


class TaskContext(BaseModel):
    task_domain: str
    task_field: str
    task_type: str
    num_metrics: int = NUM_METRICS_DEFAULT
    min_sources_per_metric: int = MIN_SOURCES_DEFAULT

    @field_validator("task_domain", "task_field", "task_type")
    @classmethod
    def _validate_context_labels(cls, v: str) -> str:
        return validate_alpha_words(v)

    @field_validator("num_metrics")
    @classmethod
    def _validate_num_metrics(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("num_metrics must be between 1 and 50")
        return v

    @field_validator("min_sources_per_metric")
    @classmethod
    def _validate_min_sources(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("min_sources_per_metric must be between 1 and 20")
        return v
