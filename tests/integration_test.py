"""
Integration test for hef_metrics_generator that exercises the public API exactly
as an end user would after `pip install hef_metrics_generator`.

This test:
- Builds the agent via initialize_metric_planner_agent_executor()
- Constructs a validated payload via build_task_context()
- Invokes the agent and asserts the validated structure is returned
- Flushes the query log to disk for audit

Notes:
- This is an ONLINE integration test. It requires valid API credentials and network.
- It will be SKIPPED automatically if OPENROUTER_API_KEY or PUBMED_EMAIL is missing.
- It also sets conservative caps to avoid context overflow during CI.

Run with:
    pytest -q tests/test_integration_generation.py -s
"""

import json
import os
import pytest

from langchain_core.runnables import RunnableConfig

from hef_metrics_generator.core.generate import (
    initialize_metric_planner_agent_executor,
    build_task_context,
)
from hef_metrics_generator.logs.tool_query_logger import query_logger

REQUIRED_ENVS = ["OPENROUTER_API_KEY", "PUBMED_EMAIL"]


def _have_required_envs() -> bool:
    for k in REQUIRED_ENVS:
        if not os.getenv(k):
            return False
    return True


@pytest.mark.skipif(not _have_required_envs(), reason="Missing required envs for integration test")
def test_integration_generate_end_to_end():
    os.environ.setdefault("OPENROUTER_MAX_TOKENS", "1200")
    # os.environ.setdefault("TOOL_CAP_RESULTS", "30")
    # os.environ.setdefault("OPENALEX_CAP", "30")
    # os.environ.setdefault("SEMANTIC_SCHOLAR_CAP", "30")
    # os.environ.setdefault("ARXIV_CAP", "60")
    # os.environ.setdefault("PUBMED_CAP", "30")
    # os.environ.setdefault("DDG_CAP", "20")

    os.environ.setdefault("AGENT_MAX_RETRIES", "2")

    agent = initialize_metric_planner_agent_executor()

    payload = build_task_context(
        task_domain="medicine",
        task_field="gastroenterology",
        task_type="diagnosis",
        num_metrics=2,
        min_sources_per_metric=2,
    )

    config = RunnableConfig(configurable={"thread_id": "integration-run-001"})

    try:
        result = agent.invoke(payload, config=config)
    finally:
        query_logger.save()

    assert isinstance(result, dict), "Agent pipeline must return a dict (validated model_dump())"
    assert "constraints" in result and "metrics" in result, "Expected 'constraints' and 'metrics' keys"

    constraints = result["constraints"]
    metrics = result["metrics"]

    assert constraints["num_metrics"] == 2
    assert constraints["min_sources_per_metric"] == 2

    assert isinstance(metrics, list) and len(metrics) == 2, "Must return exactly 2 metrics"

    for m in metrics:
        assert "metric" in m and isinstance(m["metric"], str)
        assert "min" in m and "max" in m
        assert "description" in m and "relevance" in m
        assert "sources" in m and isinstance(m["sources"], list)
        assert "search_queries" in m and isinstance(m["search_queries"], list)
        assert len(m["sources"]) >= 2

    print(json.dumps({"ok": True, "metrics_returned": len(metrics)}, indent=2))
