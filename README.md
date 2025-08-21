# HEF Metrics Generator

A Python library for generating human-evaluation metrics for LLM outputs, each backed by **real sources** from
scholarly and web tools (ArXiv, PubMed, Semantic Scholar, OpenAlex, DuckDuckGo). It enforces a strict schema and **batch
constraints** (exact count, minimum sources per metric, no duplicates).

> Built for the ChatMED/HEF pipeline; suitable for research and production use.

---

## Key Features

- **Configurable output size**: return **exactly _N_ metrics** (`num_metrics`).
- **Provenance requirements**: each metric must include **≥ K sources** (`min_sources_per_metric`).
- **Strict schema** (Pydantic v2): names (letters/spaces), scales `(1,5)` or `(0,1)`, descriptions, relevance, sources (
  `title`, `url`), and the supporting `search_queries`.
- **Multi-tool retrieval** (LangChain tools):
    - `arxiv_search` (ArXiv)
    - `pubmed_search` (PubMed)
    - `semantic_scholar_search` (Semantic Scholar)
    - `openalex_search` (OpenAlex)
    - `web_search` (DuckDuckGo)
- **Query discipline**: agent is instructed to issue **≥ 20 queries** across tools before emitting metrics (prevents
  premature/under-sourced outputs).
- **Validation helpers**: batch validator ensures exact metric count, per-metric minimum sources, no duplicate metric
  names, and no duplicate sources per metric.
- **Query logging**: all tool queries can be saved for audit/repro.

---

## Installation

### From wheel (local dist)

```bash
pip install hef_metrics_generator-0.1.0-py3-none-any.whl
```

> Python **3.9–3.10** supported. (3.11 may work but is not guaranteed; see FAQ.)

---

## Configuration (Environment)

| Variable                   | Required | Purpose                             | Example / Default                        |
|----------------------------|----------|-------------------------------------|------------------------------------------|
| `OPENROUTER_API_KEY`       | **Yes**  | LLM access (OpenRouter)             | `sk-.....`                               |
| `OPENROUTER_BASE_URL`      | No       | OpenRouter base URL                 | `https://openrouter.ai/api/v1` (default) |
| `OPENROUTER_MODEL`         | No       | Model name                          | `openai/gpt-4o` (default)                |
| `PUBMED_EMAIL`             | **Yes**  | Required by PubMed Entrez API       | `your.name@example.org`                  |
| `SEMANTIC_SCHOLAR_API_KEY` | No       | Higher Semantic Scholar limits      | *(unset ok)*                             |
| `OPENALEX_EMAIL`           | No       | Recommended by OpenAlex for contact | *(unset ok)*                             |

Create a `.env` or export shell variables before running.

---

## Quickstart

```python
import os, json
from hef-metrics-generator.core.generate import initialize_metric_planner_agent_executor
from langchain_core.runnables import RunnableConfig

# 1) Ensure env is set (example values)
os.environ.setdefault("OPENROUTER_API_KEY", "<your_key>")
os.environ.setdefault("PUBMED_EMAIL", "you@example.org")

# 2) Build the agent
agent = initialize_metric_planner_agent_executor()

# 3) Prepare the payload (num_metrics and min_sources_per_metric are configurable!)
input_payload = {
    "task_domain": "medicine",  # letters/spaces only
    "task_field": "neurology",  # letters/spaces only
    "task_type": "patient education",  # letters/spaces only
    "num_metrics": 10,  # exact number of metrics (1–50)
    "min_sources_per_metric": 5  # minimum sources required per metric (1–20)
}

# (Optional) thread id for tracing/debug
config = RunnableConfig(configurable={"thread_id": "run-001"})

# 4) Run the agent
response = agent.invoke(input_payload, config=config)

# 5) The agent now guarantees **JSON-only output** (array of metrics)
metrics = json.loads(response)

print(f"Got {len(metrics)} metrics")
print(json.dumps(metrics[0], indent=2))  # show the first metric
```

---

## Output Schema

Each metric object strictly follows:

```json
{
  "metric": "letters and spaces only",
  "min": 0,
  "max": 1,
  "description": "what this metric evaluates",
  "relevance": "why it matters for this task",
  "sources": [
    {
      "title": "Paper Title",
      "url": "https://..."
    }
  ],
  "search_queries": [
    "query 1",
    "query 2",
    "..."
  ]
}
```

- Only two scales are allowed: `(1,5)` or `(0,1)`.
- `metric` must be letters/spaces (no digits/punctuation).
- `sources` are title+URL pairs extracted directly from tools (no fabrication).
- `search_queries` capture the actual queries used for that metric.

---

## Programmatic Validation (Recommended)

Use the provided Pydantic models to **enforce the batch constraints** (exact count and per-metric min sources) and
sanity checks (no duplicates):

```python
from hef_metrics_generator.schemas.output import MetricsOutput, MetricsConstraints

constraints = MetricsConstraints(
    num_metrics=10,
    min_sources_per_metric=5,
)

validated = MetricsOutput.model_validate({
    "metrics": metrics,  # parsed list from the agent
    "constraints": constraints
})

# If invalid, Pydantic raises; otherwise you get a normalized object:
print(len(validated.metrics))  # == 10
```

---

## How It Works

- `core/generate.py` builds a LangChain **tool-calling agent** with a domain-aware system prompt:
    - Must perform **≥ 20 queries** across tools before emitting results.
    - Adapts queries per tool (e.g., PubMed for clinical terms, ArXiv for benchmarks).
    - Only emits JSON.
- Tools (in `tools/`): return structured lists of `{title, url}` items.
- Optional LLM-assisted **source relevance filter** lives in `validators/tool_validator.py`.
- A lightweight **query logger** stores all tool queries for audit/repro (`logs/tool_query_logger.py`).

---

## FAQ

**Q: Why only `(1,5)` and `(0,1)` scales?**  
A: These are enforced to standardize human evaluations across tasks and domains.

**Q: What happens if the agent emits < N metrics?**  
A: Validation will fail (`MetricsOutput`), and you should rerun or adjust constraints.

**Q: Can I add new tools?**  
A: Yes—drop a file in `tools/`, register in `tool_registry.py`, and include in the planner.

**Q: Python 3.11+?**  
A: Not officially pinned yet. You may succeed, but dependency compatibility isn’t guaranteed.

## Acknowledgments

This library is developed by **Ana Todorovska, PhD candidate**, at the **Faculty of Computer Science and Engineering,
Ss. Cyril and Methodius University in Skopje**, under the **ChatMED project (Horizon Europe, ID: 101159214)**.
