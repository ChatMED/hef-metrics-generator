"""
Metric Planner Agent for hef_metrics_generator.

This module defines the initialization of the LangChain agent responsible
for planning and generating human evaluation metrics.
It configures:
- The system prompt and behavior rules
- Integration of all registered external search tools
- Validation requirements for metrics (count, sources, format)
- LLM-based source relevance validation
- Structured-output extraction using the CANONICAL Metric schema
- Robust JSON-only extraction fallback (if structured outputs unsupported)
"""

import logging
import json
import time
import os
from typing import List, Dict, Any, Union

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate

from hef_metrics_generator.core.llm_provider import get_llm
from hef_metrics_generator.logs.default_logger import configure_logging
from hef_metrics_generator.tools.tool_registry import ALL_TOOLS

from hef_metrics_generator.schemas.output import MetricsOutput, MetricsConstraints
from hef_metrics_generator.schemas.context import TaskContext
from hef_metrics_generator.schemas.metric import Source, Metric
from hef_metrics_generator.validators.tool_validator import validate_source_with_llm

from pydantic import BaseModel

configure_logging(level=logging.DEBUG)
logger = logging.getLogger("metric_planner")

USE_STRUCTURED_EXTRACTION = os.getenv("USE_STRUCTURED_EXTRACTION", "1") == "1"


class _StructWrapper(BaseModel):
    metrics: List[Metric]


def _extract_json_array_str(raw: str) -> str:
    """
    Robustly extract a top-level JSON array string from raw model output.
    Handles:
      - raw array:           [ ... ]
      - fenced code blocks:  ```json\n[ ... ]\n```
      - prose + array:       "here is output" ... [ ... ] ... "thanks"
    """
    s = (raw or "").strip()
    if s.startswith("[") and s.endswith("]"):
        return s

    if "```" in s:
        parts = s.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            block_lines = block.splitlines()
            if block_lines:
                first = block_lines[0].strip().lower()
                if first in ("json", "json5", "jsonc"):
                    block = "\n".join(block_lines[1:])
            block = block.strip()
            if block.startswith("[") and block.endswith("]"):
                return block

    start = s.find("[")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for j in range(start, len(s)):
            ch = s[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:j + 1].strip()
                        if candidate.startswith("[") and candidate.endswith("]"):
                            return candidate

    raise ValueError("Could not extract a top-level JSON array from model output")


def _normalize_metrics_list(
        metrics_list: List[Union[Metric, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Normalize a list of Metric objects or dicts into plain dicts
    (with primitive URL strings), suitable for downstream validation and JSON.
    """
    normalized: List[Dict[str, Any]] = []
    for m in metrics_list:
        if isinstance(m, Metric):
            normalized.append(m.model_dump())
        else:
            normalized.append(m)
    return normalized


def _structured_extract_metrics(raw_text: str, llm) -> List[Dict[str, Any]]:
    """
    Use LLM structured output to extract ONLY the metrics array using the
    canonical Metric schema. If extraction succeeds, return list[dict].
    """
    structured = llm.with_structured_output(_StructWrapper)
    prompt = (
        "Extract the list of metrics from the following text into the schema "
        "`{ metrics: [ { metric, min, max, description, relevance, sources[{title,url}], search_queries[] } ] }`.\n"
        "You MUST NOT fabricate values; only include items that are clearly present in the text. "
        "If none are present, return an empty list.\n\n"
        "TEXT:\n"
        f"{raw_text}"
    )
    wrapper: _StructWrapper = structured.invoke(prompt)
    return _normalize_metrics_list(wrapper.metrics)


def _assert_min_max_combo(metrics_list: List[Dict[str, Any]]) -> None:
    """
    Explicit guard: For each metric, (min,max) must be either (0,1) or (1,5).
    This is in addition to the canonical Metric schema's validator,
    providing an early, human-readable error if violated.
    """
    for idx, m in enumerate(metrics_list, start=1):
        min_v = m.get("min")
        max_v = m.get("max")
        combo = (min_v, max_v)
        if combo not in {(0, 1), (1, 5)}:
            raise ValueError(
                f"Metric #{idx} ('{m.get('metric', '<unnamed>')}') has invalid min/max combo {combo}; "
                "allowed combinations are (0,1) or (1,5)."
            )


def _validate_and_filter_metrics(
        metrics_list: List[Dict[str, Any]],
        inputs: Dict[str, Any],
        llm,
) -> List[Dict[str, Any]]:
    """
    Apply LLM-based source filtering and batch validation, then return JUST the metrics list.
    Also enforce min/max combo constraint explicitly.
    """
    filtered_metrics: List[Dict[str, Any]] = []
    for metric in metrics_list:
        filtered_sources = []
        for src in (metric.get("sources") or []):
            try:
                validated = validate_source_with_llm(
                    Source(**src), llm, inputs["task_domain"]
                )
                if validated:
                    filtered_sources.append(src)
            except Exception:
                continue
        if filtered_sources:
            metric["sources"] = filtered_sources
            filtered_metrics.append(metric)

    _assert_min_max_combo(filtered_metrics)

    constraints = MetricsConstraints(
        num_metrics=inputs["num_metrics"],
        min_sources_per_metric=inputs["min_sources_per_metric"],
    )
    _ = MetricsOutput(constraints=constraints, metrics=filtered_metrics)
    return filtered_metrics


def initialize_metric_planner_agent_executor() -> Runnable:
    llm = get_llm()

    system = """
    You are an expert in the domain of {task_domain} and a researcher specializing
    in evaluating large language models (LLMs) using human evaluation in this domain.

    You are given an evaluation task with the following context:
        - Task type: {task_type}
        - Task field: {task_field}
        - Task domain: {task_domain}
        - Number of metrics to return EXACTLY: {num_metrics}
        - Minimum sources required per metric: {min_sources_per_metric}

    Your goal is to generate exactly {num_metrics} metrics for human evaluation of LLM responses related to this task.

    Tools available (MUST use them; never fabricate sources):
        - arxiv_search (ArXiv)
        - pubmed_search (PubMed)
        - semantic_scholar_search (Semantic Scholar)
        - openalex_search (OpenAlex)
        - web_search (DuckDuckGo; fallback only)

    Each tool returns a JSON list of:
      {{"title": "...", "url": "https://..."}}
    You MUST extract titles and URLs exactly from tool output. Do NOT modify them.

    Strategy (strict):
    1) Issue at least 20 keyword-style queries across tools BEFORE proposing any metric.
       Vary specificity using task_type/task_field/task_domain synonyms.
       Avoid full sentences. Examples:
         - "LLM evaluation {task_type} {task_domain}"
         - "human eval generative AI {task_field}"
         - "benchmark rubric {task_domain}"

    2) Priority by domain:
       - If healthcare/biomed/psychology → prioritize pubmed_search, then arxiv_search, then semantic_scholar_search / openalex_search. Use web_search only if needed.
       - If AI/CS → prioritize arxiv_search, then semantic_scholar_search, then openalex_search; web_search only if needed.
       - Otherwise (general) → semantic_scholar_search + openalex_search first, then arxiv_search; web_search only if needed.

    3) Adapt queries by tool:
        - arxiv_search: scientific phrasing (benchmarks, human eval, reliability)
        - pubmed_search: clinical/empirical phrasing (patient-facing, clinical outcomes)
        - semantic_scholar_search / openalex_search: broad scholarly phrasing; filter by relevance
        - web_search: ONLY if scholarly tools fail to cover an angle

    4) Dynamically rephrase if results are poor:
        - Try synonyms (“assessment”, “benchmarking”, “human rating”)
        - Back off specificity gradually
        - Stay within the given domain/field/type

    5) Do NOT emit metrics until you have issued >= 20 queries total across tools.

    6) For each candidate metric:
        - Include the list of search queries you used to support it
        - Cite ALL valid sources you rely on
        - Discard the metric if it has fewer than {min_sources_per_metric} valid sources

    7) Each source MUST:
        - Clearly relate to human evaluation of LLMs (or very close analogue)
        - Be relevant to the task type, field, or domain
        - Include BOTH a title and a URL (from tool output)

    8) OUTPUT FORMAT RULES (MANDATORY):
        - Return ONLY a raw JSON array, no prose, no markdown, no explanation.
        - JSON must be a list of exactly {num_metrics} objects.
        - Each object must follow this schema exactly:
          {{
            "metric": "letters and spaces only",
            "min": 0 or 1,
            "max": 1 or 5,
            "description": "what this metric evaluates",
            "relevance": "why it matters for this task",
            "sources": [
              {{"title": "Paper Title 1", "url": "https://..."}},
              {{"title": "Paper Title 2", "url": "https://..."}}
            ],
            "search_queries": [
              "query used to find source 1",
              "query used to find source 2"
            ]
          }}

    IMPORTANT:
    - Return EXACTLY {num_metrics} metrics.
    - Each metric must have at least {min_sources_per_metric} sources.
    - For every metric, the only valid (min,max) pairs are EXACTLY (1,5) OR (0,1). Any other pair is invalid.
    - Never invent titles or URLs; only use tool output.
    - If multiple plausible metrics exist, keep the {num_metrics} strongest and drop the rest.
    - If you output anything other than a valid JSON list, your response will be discarded and retried.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user",
         "Task type: {task_type}\n"
         "Task field: {task_field}\n"
         "Task domain: {task_domain}\n"
         "Number of metrics: {num_metrics}\n"
         "Minimum sources per metric: {min_sources_per_metric}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    base_llm = get_llm()
    agent = create_tool_calling_agent(base_llm, ALL_TOOLS, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        handle_parsing_errors=True
        # Optionally: max_iterations=12, early_stopping_method="generate"
    )

    def _postprocess(payload: Dict[str, Any]) -> str:
        """
        payload: {"original_inputs": <inputs dict>, "agent_output": <raw from agent>}
        Returns: JSON string of the validated metrics array ONLY.
        """
        inputs = payload["original_inputs"]
        output = payload["agent_output"]

        max_retries = int(os.getenv("AGENT_MAX_RETRIES", "3"))
        llm_for_validation = get_llm()

        for attempt in range(1, max_retries + 1):
            try:
                raw = output if isinstance(output, str) else output.get("output", output)
                if not isinstance(raw, str):
                    raise ValueError(f"Unexpected agent output type: {type(raw)}")

                metrics_list: List[Dict[str, Any]] = []
                structured_ok = False
                if USE_STRUCTURED_EXTRACTION:
                    try:
                        metrics_list = _structured_extract_metrics(raw, llm_for_validation)
                        structured_ok = True
                    except Exception as se:
                        logger.warning(f"[structured_extract] Falling back to raw JSON extraction: {se}")

                if not structured_ok:
                    array_str = _extract_json_array_str(raw)
                    metrics_list = json.loads(array_str)
                    if not isinstance(metrics_list, list):
                        raise ValueError("Agent output is not a JSON list")
                    metrics_list = _normalize_metrics_list(metrics_list)

                filtered = _validate_and_filter_metrics(metrics_list, inputs, llm_for_validation)

                return json.dumps(filtered, ensure_ascii=False)

            except Exception as e:
                if attempt == max_retries:
                    raise ValueError(
                        f"Agent failed to produce valid JSON after {max_retries} attempts: {e}"
                    )
                time.sleep(0.5)
                output = agent_executor.invoke(inputs)

        raise RuntimeError("Unreachable: retries exhausted")

    pipeline: Runnable = (
            {
                "original_inputs": RunnablePassthrough(),
                "agent_output": agent_executor,
            }
            | RunnableLambda(_postprocess)
    )

    return pipeline


def build_task_context(
        task_domain: str,
        task_field: str,
        task_type: str,
        num_metrics: int,
        min_sources_per_metric: int,
) -> dict:
    """
    Helper: Build and validate a TaskContext from user inputs.
    Returns a dict usable by the agent.
    """
    ctx = TaskContext(
        task_domain=task_domain,
        task_field=task_field,
        task_type=task_type,
        num_metrics=num_metrics,
        min_sources_per_metric=min_sources_per_metric,
    )
    return ctx.model_dump()
