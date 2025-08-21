"""
Metrics output schema for hef_metrics_generator.

Defines the container model for a batch of metrics plus the constraints
used to validate them (exact count and per-metric minimum sources).
Also performs cross-metric sanity checks (e.g., no duplicate metric names).
"""

from __future__ import annotations
from typing import List, Tuple, Set
from pydantic import BaseModel, field_validator, model_validator

from hef_metrics_generator.utils.constants import NUM_METRICS_DEFAULT, MIN_SOURCES_DEFAULT


class MetricsConstraints(BaseModel):
    """
    Constraints to validate a metrics batch.
    These are NOT the domain/field/type task context — those live elsewhere.
    """
    num_metrics: int = NUM_METRICS_DEFAULT
    min_sources_per_metric: int = MIN_SOURCES_DEFAULT

    @field_validator("num_metrics")
    @classmethod
    def _validate_num_metrics(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_metrics must be positive")
        return v

    @field_validator("min_sources_per_metric")
    @classmethod
    def _validate_min_sources(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sources_per_metric must be >= 1")
        return v


class MetricsOutput(BaseModel):
    """
    Container for all metrics produced by the agent.
    Validates both structural correctness (via Metric schema)
    and batch-level constraints (via MetricsConstraints).
    """
    constraints: MetricsConstraints
    metrics: List[Metric]

    @model_validator(mode="after")
    def _validate_batch(self) -> "MetricsOutput":
        expected = self.constraints.num_metrics
        actual = len(self.metrics)
        if actual != expected:
            raise ValueError(f"Expected {expected} metrics, got {actual}")

        min_src = self.constraints.min_sources_per_metric
        for m in self.metrics:
            if len(m.sources) < min_src:
                raise ValueError(
                    f"Metric '{m.metric}' has {len(m.sources)} sources, "
                    f"but requires at least {min_src}"
                )

        names: List[str] = [m.metric for m in self.metrics]
        dupes = _find_duplicates(names)
        if dupes:
            raise ValueError(f"Duplicate metric names found: {sorted(dupes)}")

        for m in self.metrics:
            seen_pairs: Set[Tuple[str, str]] = set()
            for s in m.sources:
                key = (s.title.strip(), str(s.url).strip())
                if key in seen_pairs:
                    raise ValueError(
                        f"Metric '{m.metric}' has duplicate source: {key}"
                    )
                seen_pairs.add(key)

        return self


def _find_duplicates(items: List[str]) -> Set[str]:
    """
    Return the set of duplicated items in the given list.
    """
    seen: Set[str] = set()
    dupes: Set[str] = set()
    for it in items:
        if it in seen:
            dupes.add(it)
        else:
            seen.add(it)
    return dupes
